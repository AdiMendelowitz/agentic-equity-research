"""Pydantic v2 schema for the artifact of record, manifest, and run-status sidecar.

Canonical schema: ``run_pipeline`` writes one
``BriefArtifactV1`` per ticker (embedding the full ``InvestmentBrief`` plus a list of
``SignalRecord``), a ``ManifestV1`` (``latest.json``) mapping tickers to their current file, and
a ``RunStatusV1`` sidecar. The dashboard and simulator become pure readers; neither invokes the
graph.

Linkage: each ``SignalRecord.signal_id`` is derived ONCE in
``run_pipeline`` via a local sha256 of ticker, ``data_asof_date`` and a fixed ``synthesis`` token.
``build_synthesis_signal_record`` is the single derivation point.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from aer.models import InvestmentBrief
from aer.paths import PROJECT_ROOT
from aer.utils.atomic_io import atomic_write_json, read_json

logger = logging.getLogger(__name__)


def derive_signal_id(ticker: str, date: str, run_id: str) -> str:
    return hashlib.sha256(f"{ticker}|{date}|{run_id}".encode()).hexdigest()


Direction = Literal["BUY", "HOLD", "SELL", "AVOID"]

ARTIFACT_DIR_ENV = "AER_ARTIFACT_DIR"
# Resolved from aer.paths: parents[2] of this file is src/, not the project root.
# CWD-relative default was a silent-scatter hazard when the working directory varies.
_DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "briefs"
MANIFEST_FILENAME = "latest.json"
RUN_STATUS_FILENAME = "run_status.json"
_ILLEGAL_FILENAME_CHARS = ':<>"/\\|?*'


class SignalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    agent: str
    direction: Direction
    data_asof_date: date
    confidence: float
    score: float | None = None
    signal_id: str


class BriefArtifactV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    ticker: str
    run_id: str
    git_sha: str
    generated_at_utc: datetime
    data_asof_date: date
    intended_entry_date: date
    watchlist_sha256: str
    models_used: dict[str, str]
    professor_critique: str | None = None
    signals: list[SignalRecord]
    brief: InvestmentBrief


class TickerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    run_id: str
    generated_at_utc: datetime
    data_asof_date: date
    intended_entry_date: date


class ManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    last_run_id: str
    generated_at_utc: datetime
    watchlist_sha256: str
    tickers: dict[str, TickerEntry]


class RunStatusV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    generated_at_utc: datetime
    watchlist_sha256: str
    per_ticker: dict[str, Literal["OK", "FAILED"]]
    per_ticker_exception: dict[str, str]
    exit_code: int
    # extension fields (additive, default-safe; old on-disk run_status.json files and
    # existing readers stay compatible).
    fatal: bool = False
    fatal_error: str | None = None


# ── helpers ───────────────────────────────────────────────────────────────────
def artifact_dir() -> Path:
    """Artifact output directory; ``AER_ARTIFACT_DIR`` overrides for test isolation."""
    env = os.getenv(ARTIFACT_DIR_ENV)
    if env:
        path = Path(env)
    else:
        logger.warning("AER_ARTIFACT_DIR not set; falling back to default %s", _DEFAULT_ARTIFACT_DIR)
        path = _DEFAULT_ARTIFACT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize(name: str) -> str:
    return "".join("_" if ch in _ILLEGAL_FILENAME_CHARS else ch for ch in name)


def artifact_filename(ticker: str, utc_date: str, run_id_value: str) -> str:
    """``brief_<ticker>_<utc_date>_<run_id_sanitized>.json`` (colon-safe on Windows)."""
    return f"brief_{_sanitize(ticker.upper())}_{_sanitize(utc_date)}_{_sanitize(run_id_value)}.json"


def watchlist_sha256(tickers: list[str]) -> str:
    """Order-independent hash of the watchlist for cross-file provenance."""
    canonical = "\n".join(sorted(t.strip().upper() for t in tickers))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_synthesis_signal_record(brief: InvestmentBrief, data_asof_date: date) -> SignalRecord:
    """Build the synthesis ``SignalRecord`` with the parity ``signal_id`` (keyed on data_asof_date).

    This is the single signal_id derivation point.
    """
    sig_id = derive_signal_id(brief.ticker, data_asof_date.isoformat(), "synthesis")
    return SignalRecord(
        ticker=brief.ticker,
        agent="synthesis",
        direction=brief.recommendation,
        data_asof_date=data_asof_date,
        confidence=brief.conviction,
        score=brief.conviction,
        signal_id=sig_id,
    )


def load_artifact(path: Path) -> BriefArtifactV1:
    return BriefArtifactV1.model_validate(read_json(Path(path)))


def write_artifact_atomic(artifact: BriefArtifactV1, path: Path) -> None:
    BriefArtifactV1.model_validate(artifact.model_dump())  # re-validate before persisting
    atomic_write_json(Path(path), artifact.model_dump(mode="json"))


def load_manifest(path: Path) -> ManifestV1 | None:
    p = Path(path)
    if not p.exists():
        return None
    return ManifestV1.model_validate(read_json(p))


def merge_and_write_manifest_atomic(
    prior: ManifestV1 | None,
    updates: dict[str, TickerEntry],
    watchlist: list[str],
    watchlist_hash: str,
    run_id: str,
    *,
    path: Path | None = None,
    generated_at_utc: datetime | None = None,
    verify_dir: Path | None = None,
) -> ManifestV1:
    """Merge a prior manifest with this run's successful entries, then write it atomically.

    Semantics: prior seeds; tickers absent from ``watchlist`` are dropped;
    a fresh entry replaces the prior; a failed ticker (in watchlist, no update) retains its prior
    entry; an entry is never kept if its file is missing (checked when ``verify_dir`` is given).
    """
    allowed = {t.strip().upper() for t in watchlist}
    tickers: dict[str, TickerEntry] = {}
    if prior is not None:
        tickers = {t: e for t, e in prior.tickers.items() if t.upper() in allowed}
    tickers.update(updates)
    if verify_dir is not None:
        tickers = {t: e for t, e in tickers.items() if (Path(verify_dir) / e.file).exists()}

    manifest = ManifestV1(
        last_run_id=run_id,
        generated_at_utc=generated_at_utc or datetime.now(),
        watchlist_sha256=watchlist_hash,
        tickers=tickers,
    )
    target = Path(path) if path is not None else (artifact_dir() / MANIFEST_FILENAME)
    atomic_write_json(target, manifest.model_dump(mode="json"))
    return manifest


def load_run_status(path: Path) -> RunStatusV1 | None:
    p = Path(path)
    if not p.exists():
        return None
    return RunStatusV1.model_validate(read_json(p))


def write_run_status_atomic(status: RunStatusV1, path: Path) -> None:
    atomic_write_json(Path(path), status.model_dump(mode="json"))
