"""Single compute entry point for the B2b pipeline — ``python -m aer.run_pipeline``.

Producer of record: runs the graph once per watchlist ticker, derives each signal_id ONCE per run
(keyed on the run's ``data_asof_date``), hands that id to the persisted artifact, and writes the per-ticker artifact, the manifest (``latest.json``), and the
run-status sidecar (``run_status.json``) — all via atomic writes under a single-instance lock.
The dashboard and simulator read these artifacts; neither invokes the graph.

Scheduling: 08:00 IL == 01:00-02:00 ET (pre-open), DST-safe (IL<->ET offset is 6h or 7h, never 8h).
At that pre-open time ``last_completed_session`` returns the prior trading day, so ``data_asof_date``
= last completed session and ``intended_entry_date`` = the next trading day (= today), which is when
the simulator's entry gate fires. ``data_asof_date`` is a RUN property: it is computed once and every
ticker in the run shares it (and one ``run_id``).

Exit codes (this module; the SIMULATOR's 0/2/3 in daily_runner mean different things and live in a
different status file):
    0  every watchlist ticker produced a schema-valid persisted artifact and the manifest updated
    1  partial — at least one ticker succeeded and at least one failed
    2  total failure — no artifact written, empty watchlist, or a fatal/unexpected error

Session A note: ``graph.invoke_pipeline`` is imported lazily inside ``_run_one_ticker`` so importing
this module never pulls the heavy graph/LLM stack; the old graph invoke path still exists
until Session B.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from aer.paths import ENV_FILE as _ENV_PATH
from aer.utils.artifact_schema import (
    MANIFEST_FILENAME,
    RUN_STATUS_FILENAME,
    BriefArtifactV1,
    RunStatusV1,
    TickerEntry,
    artifact_dir,
    artifact_filename,
    build_synthesis_signal_record,
    load_manifest,
    load_run_status,
    merge_and_write_manifest_atomic,
    watchlist_sha256,
    write_artifact_atomic,
    write_run_status_atomic,
)
from aer.utils.constants import GIT_TIMEOUT_SECONDS, PIPELINE_LOCK_STALE_HOURS, PIPELINE_LOG_PATH, SYNTHESIS_MODEL
from aer.utils.locking import acquire_lock, release_lock
from aer.utils.pipeline_heartbeat import write_heartbeat
from aer.utils.trading_calendar import is_trading_day, last_completed_session, next_trading_day

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_LOCK_FILENAME = ".run_pipeline.lock"
_UNKNOWN_SHA = "unknown"

_LLM_KEY_NAMES = ("OPENROUTER_API_KEY", "NVIDIA_NIM_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY")
LLM_USAGE_FILENAME = "llm_usage.jsonl"


def _derive_run_id(run_start_utc_iso: str) -> str:
    return f"{run_start_utc_iso}_{uuid.uuid4().hex[:8]}"


def load_runtime_env(env_path: Path = _ENV_PATH) -> dict[str, bool]:
    """Load.env into the process for the scheduled run and report which LLM keys are present.

    Called from the entry point only, never at import (the lesson). A variable already in the
    environment wins over the file. The return value holds key names and booleans, never values.
    Must run before anything imports aer.utils.llm_config, which reads the keys at import time.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv(env_path, override=False)
    return {name: bool(os.environ.get(name)) for name in _LLM_KEY_NAMES}


def _git_sha() -> str:
    """Best-effort short git SHA of HEAD; ``'unknown'`` if git is unavailable."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
        return r.stdout.strip() if r.returncode == 0 else _UNKNOWN_SHA
    except Exception:  # pragma: no cover - defensive
        return _UNKNOWN_SHA


def _run_one_ticker(
    ticker: str,
    run_id_value: str,
    git_sha: str,
    watchlist_hash: str,
    data_asof_date: date,
    intended_entry_date: date,
) -> BriefArtifactV1:
    """Compute one ticker end-to-end and persist its artifact; return the artifact.

    Run-global values (``run_id_value``, ``data_asof_date``, ``intended_entry_date``) are computed
    once in ``main`` and passed in, so every ticker in a run shares one as-of date and run id.

    """
    from aer.graph import invoke_pipeline  # lazy: avoid importing the graph/LLM stack at module load

    brief, meta = invoke_pipeline(ticker)

    signal = build_synthesis_signal_record(brief, data_asof_date)
    synthesis_model = meta.get("synthesis_model_used")
    if not synthesis_model:
        logger.warning("synthesis_model_used missing from meta; falling back to SYNTHESIS_MODEL constant")
        synthesis_model = SYNTHESIS_MODEL

    artifact = BriefArtifactV1(
        ticker=ticker.upper(),
        run_id=run_id_value,
        git_sha=git_sha,
        generated_at_utc=datetime.now(UTC),
        data_asof_date=data_asof_date,
        intended_entry_date=intended_entry_date,
        watchlist_sha256=watchlist_hash,
        models_used={"synthesis": synthesis_model},
        signals=[signal],
        brief=brief,
    )

    write_artifact_atomic(
        artifact, artifact_dir() / artifact_filename(ticker, data_asof_date.isoformat(), run_id_value)
    )
    return artifact


def _write_usage_summary(run_id_value: str | None) -> None:
    """Append this run's metered LLM tokens (usage_meter) as one line beside the artifacts; never raises."""
    from aer.utils import usage_meter

    try:
        summary = usage_meter.write_run_summary(artifact_dir() / LLM_USAGE_FILENAME, run_id_value or "unknown-fatal")
        logger.info("run_pipeline: LLM usage %s", summary["totals"])
    except Exception as exc:  # noqa: BLE001 - metering must never change a run's outcome
        logger.warning("run_pipeline: LLM usage summary not written: %s", exc)


def main(argv: list[str] | None = None, *, now_et: datetime | None = None) -> int:
    """Run the pipeline over the watchlist; write artifacts, manifest, and status. Return exit code."""
    from aer.utils.watchlist import load_watchlist  # local import (mirrors daily_runner; avoids UI import at load)

    lock_path = artifact_dir() / _LOCK_FILENAME
    acquire_lock(lock_path, PIPELINE_LOCK_STALE_HOURS)
    run_id_value: str | None = None
    watchlist_hash: str | None = None
    beat: dict = {}  # heartbeat context, set once the run_id exists
    exit_code = 2
    per_ticker: dict[str, str] = {}
    try:
        watchlist = load_watchlist()
        if not watchlist:
            logger.error("run_pipeline: empty watchlist — nothing to run")
            return 2
        watchlist_hash = watchlist_sha256(watchlist)
        logger.info("run_pipeline: %d tickers, watchlist_sha256=%s...", len(watchlist), watchlist_hash[:12])

        # Run-global values: one clock read, one as-of date, one run_id shared by every ticker.
        now_et = now_et or datetime.now(_ET)
        data_asof_date = last_completed_session(now_et)
        intended_entry_date = next_trading_day(data_asof_date)
        run_start_utc_iso = datetime.now(UTC).isoformat()
        git_sha = _git_sha()
        if git_sha == _UNKNOWN_SHA:
            logger.error(
                "run_pipeline: git SHA unavailable — artifacts will lack code provenance and are "
                "NOT admissible to the 90-day record until re-run from a clean commit"
            )
        run_id_value = _derive_run_id(run_start_utc_iso)
        beat = {"run_id": run_id_value, "total": len(watchlist)}
        write_heartbeat(**beat, done=0, last_ticker=None, state="running")

        # (Adi 2026-09-29): a run dated on a closed day writes opinions that never get a forward return, and
        # spends the free LLM budget on them. The trigger stays daily so run_status.json keeps refreshing (the 30 h
        # dead-man's switch); a closed day exits 0 with an empty per_ticker map. PIPELINE_RUN_NON_TRADING_DAYS=1
        # overrides it for a manual run. now_et is the scheduled start time, so the gate covers the 08:00 IL task;
        # a manual run in the IL small hours of a Saturday still sees an ET Friday.
        if os.environ.get("PIPELINE_RUN_NON_TRADING_DAYS") != "1" and not is_trading_day(now_et.date()):
            logger.info(
                "run_pipeline: %s is not an NYSE trading day; no ticker is run and no opinion written", now_et.date()
            )
            exit_code = 0
            status_path = artifact_dir() / RUN_STATUS_FILENAME
            prior = load_run_status(status_path)
            now_utc = datetime.now(UTC)
            # Carry the last trading-day evidence (failed tickers, run_id, manifest link) forward; only the
            # freshness stamp moves. With no readable prior status, an empty one is written.
            write_run_status_atomic(
                (
                    prior.model_copy(update={"generated_at_utc": now_utc})
                    if prior is not None
                    else RunStatusV1(
                        run_id=run_id_value,
                        generated_at_utc=now_utc,
                        watchlist_sha256=watchlist_hash,
                        per_ticker={},
                        per_ticker_exception={},
                        exit_code=0,
                    )
                ),
                status_path,
            )
            return 0

        per_ticker_exception: dict[str, str] = {}
        updates: dict[str, TickerEntry] = {}

        for ticker in watchlist:
            tkr = ticker.upper()
            try:
                artifact = _run_one_ticker(
                    tkr, run_id_value, git_sha, watchlist_hash, data_asof_date, intended_entry_date
                )
                per_ticker[tkr] = "OK"
                updates[tkr] = TickerEntry(
                    file=artifact_filename(tkr, data_asof_date.isoformat(), run_id_value),
                    run_id=run_id_value,
                    generated_at_utc=artifact.generated_at_utc,
                    data_asof_date=data_asof_date,
                    intended_entry_date=intended_entry_date,
                )
            except Exception as e:
                per_ticker[tkr] = "FAILED"
                per_ticker_exception[tkr] = f"{type(e).__name__}: {e}"
                logger.warning("run_pipeline: ticker %s FAILED: %s", tkr, e)
            write_heartbeat(**beat, done=len(per_ticker), last_ticker=tkr, state="running")

        ok_count = sum(1 for v in per_ticker.values() if v == "OK")
        failed_count = len(per_ticker) - ok_count
        if ok_count and not failed_count:
            exit_code = 0
        elif ok_count and failed_count:
            exit_code = 1
        else:
            exit_code = 2

        prior = load_manifest(artifact_dir() / MANIFEST_FILENAME)
        merge_and_write_manifest_atomic(
            prior=prior,
            updates=updates,
            watchlist=watchlist,
            watchlist_hash=watchlist_hash,
            run_id=run_id_value,
            verify_dir=artifact_dir(),
        )

        status = RunStatusV1(
            run_id=run_id_value,
            generated_at_utc=datetime.now(UTC),
            watchlist_sha256=watchlist_hash,
            per_ticker=per_ticker,
            per_ticker_exception=per_ticker_exception,
            exit_code=exit_code,
        )
        write_run_status_atomic(status, artifact_dir() / RUN_STATUS_FILENAME)
        return exit_code
    except Exception as fatal:
        logger.critical("run_pipeline: fatal error, aborting run: %s", fatal, exc_info=True)
        try:
            degraded_status = RunStatusV1(
                run_id=run_id_value or "unknown-fatal",
                generated_at_utc=datetime.now(UTC),
                watchlist_sha256=watchlist_hash or "unknown",
                per_ticker={},
                per_ticker_exception={},
                exit_code=2,
                fatal=True,
                fatal_error=f"{type(fatal).__name__}: {fatal}",
            )
            write_run_status_atomic(degraded_status, artifact_dir() / RUN_STATUS_FILENAME)
        except Exception as status_write_error:
            logger.error(
                "run_pipeline: failed to write degraded run_status.json after fatal error: %s",
                status_write_error,
            )
        exit_code = 2
        return 2
    finally:
        _write_usage_summary(run_id_value)
        if beat:
            write_heartbeat(**beat, done=len(per_ticker), last_ticker=None, state="finished", exit_code=exit_code)
        release_lock(lock_path)


if __name__ == "__main__":  # pragma: no cover
    PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(PIPELINE_LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    logger.info("run_pipeline: LLM keys present (names only): %s", load_runtime_env())
    sys.exit(main())
