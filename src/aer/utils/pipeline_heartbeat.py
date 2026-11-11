"""Pipeline run heartbeat, 2026-09-25).

run_pipeline writes ``pipeline_heartbeat.json`` beside run_status.json at start, after every ticker
and at exit, so a run that dies or hangs is visible while it happens. The 2026-09-24 run stopped
mid-ticker at 20:44:02Z and nothing recorded it for ~14 h, because run_status.json is written only
at the end. ``assess`` is the single judgement used by scripts/pipeline_watchdog.py.

"Run completed" is still the pipeline exit line plus a fresh run_status.json; the heartbeat is
liveness only.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aer.utils.artifact_schema import artifact_dir
from aer.utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

HEARTBEAT_FILENAME = "pipeline_heartbeat.json"
# 2.5 x the 11.7 min maximum per-ticker gap measured on 2026-09-24 (median 10.3 min)
STALL_MINUTES = 30


def heartbeat_path() -> Path:
    return artifact_dir() / HEARTBEAT_FILENAME


def write_heartbeat(
    *, run_id: str, total: int, done: int, last_ticker: str | None, state: str, exit_code: int | None = None
) -> None:
    """Best effort: a heartbeat write must never break a pipeline run."""
    payload = {
        "run_id": run_id,
        "pid": os.getpid(),
        "state": state,
        "done": done,
        "total": total,
        "last_ticker": last_ticker,
        "exit_code": exit_code,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    try:
        path = heartbeat_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload)
    except Exception as exc:  # noqa: BLE001 - liveness must not become a failure mode
        logger.warning("pipeline heartbeat write failed: %s", exc)


def assess(
    beat: dict[str, Any] | None, *, now: datetime, pid_alive: Callable[[int], bool], stall_minutes: int = STALL_MINUTES
) -> tuple[str, str] | None:
    """Return (kind, message) for a run that needs attention, else None.

    DIED: state "running" and the owning process is gone (killed, crashed, machine restarted).
    STALLED: state "running", process alive, no heartbeat for more than ``stall_minutes``
    (a hang, or a machine that slept mid-run). A missing heartbeat is the audit's
    PIPELINE_HEARTBEAT_STALE case, not the watchdog's.
    """
    if not beat or beat.get("state") != "running":
        return None
    updated = datetime.fromisoformat(beat["updated_utc"])
    age_min = (now - updated).total_seconds() / 60
    where = f"run {beat.get('run_id')} at {beat.get('done')}/{beat.get('total')} (last {beat.get('last_ticker')})"
    if not pid_alive(int(beat.get("pid") or 0)):
        return "DIED", f"pipeline process {beat.get('pid')} is gone, {where}, last heartbeat {age_min:.0f} min ago"
    if age_min > stall_minutes:
        return "STALLED", f"no heartbeat for {age_min:.0f} min (> {stall_minutes}), process alive, {where}"
    return None
