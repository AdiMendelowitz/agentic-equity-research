"""Per-run, per-(ticker, domain) completeness manifest writer.

Persists one JSON line per (ticker, domain) status to a JSONL file after each
prefetch run so downstream phases have a join key for completeness analysis.

Statuses mirror the four observable outcomes of the prefetch driver:
 fetched - cache was cold; fetcher ran and wrote a key
 cache-hit - cache was warm; fetcher was skipped (hit_before=True)
 skipped-by-budget - spec hit its per-domain call budget limit
 errored - fetcher raised an exception or timed out

The distinction between skipped-by-budget, errored and true-zero (never fetched) is the
NaN-vs-zero discipline: a domain that was intentionally budget-limited must not be confused
with a domain that failed or was never attempted.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aer.paths import CACHE_DIR as _CACHE_DIR

_logger = logging.getLogger(__name__)

# JSONL output directory and file path.
# Located under the cache directory in prefetch_logs/.

_PREFETCH_LOGS_DIR: Path = _CACHE_DIR / "prefetch_logs"
_COMPLETENESS_JSONL: Path = _PREFETCH_LOGS_DIR / "completeness.jsonl"

# Valid status values - enforced at write time to prevent typos propagating into JSONL.
VALID_STATUSES: frozenset[str] = frozenset({"fetched", "cache-hit", "skipped-by-budget", "errored"})


def new_run_id() -> str:
    """Return a new unique run identifier (UUID4 hex string)."""
    return uuid.uuid4().hex


def record_completeness(run_id: str, rows: list[dict[str, Any]]) -> None:
    """Append one JSON line per (ticker, domain) status row to the completeness JSONL.

    Each row is expected to carry at minimum:
        ticker:  str | None  - watchlist ticker, or None for global specs
        source:  str         - SourceSpec.name
        domain:  str         - cache domain (data/agent/news/lseg)
        status:  str         - one of VALID_STATUSES

    The writer adds run_id and timestamp to every row before appending.
    Rows with an unrecognised status value are logged at WARNING and skipped.

    Parameters
    ----------
    run_id:
        Unique identifier for this prefetch run (from new_run_id()).
    rows:
        Per-(ticker, domain) status dicts assembled by run_prefetch.
    """
    _PREFETCH_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat()

    lines_written = 0
    with _COMPLETENESS_JSONL.open("a", encoding="utf-8") as fh:
        for row in rows:
            status = row.get("status")
            if status not in VALID_STATUSES:
                _logger.warning(
                    "record_completeness: unknown status=%r for source=%s ticker=%s — skipping",
                    status,
                    row.get("source"),
                    row.get("ticker"),
                )
                continue
            record: dict[str, Any] = {
                "run_id": run_id,
                "ts": ts,
                **row,
            }
            fh.write(json.dumps(record) + "\n")
            lines_written += 1

    _logger.info(
        "record_completeness: wrote %d rows to %s (run_id=%s)",
        lines_written,
        _COMPLETENESS_JSONL,
        run_id,
    )
