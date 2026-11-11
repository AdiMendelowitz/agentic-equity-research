"""Append-only recommendation log for agent feedback system.

Each agent writes one record per recommendation to:
  agents/{agent_id}/memory/recommendations.jsonl

Outcome records live in a separate file (outcomes.jsonl) written by the outcome labeler only.
"""

import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from datetime import datetime as _datetime
from pathlib import Path
from typing import Any

import portalocker

from aer.utils.constants import (
    REASONING_SUMMARY_MAX_CHARS,
    SCHEMA_VERSION,
)
from aer.utils.db import connect_db
from aer.utils.db_sync import _DB_PATH, init_db, sync_recommendation

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _agents_root() -> Path:
    return _PROJECT_ROOT / "agents"


def _reasoning_store_root() -> Path:
    env = os.environ.get("REASONING_STORE_ROOT")
    if env:
        return Path(env)
    return _agents_root()


def _assert_utc(ts: str, field: str) -> None:
    """Reject naive timestamps — must contain +00:00 or end with Z, and be valid ISO8601."""
    if not (ts.endswith("Z") or "+00:00" in ts or "+0000" in ts):
        raise ValueError(f"{field} must be UTC ISO8601 (got {ts!r})")
    try:
        _datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{field} is not a valid ISO8601 datetime (got {ts!r})")


def write_recommendation(record: dict[str, Any]) -> None:
    """Validate and append one recommendation record to the agent's JSONL log.

    Raises ValueError for any schema violation. Acquires a file lock before writing.
    """
    # Schema version
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {record.get('schema_version')!r}")

    # Required fields
    for field in ("rec_id", "agent_id", "ts_rec", "ts_eval_open", "ticker", "signal", "horizons"):
        if field not in record:
            raise ValueError(f"Missing required field: {field!r}")

    # UTC enforcement
    _assert_utc(record["ts_rec"], "ts_rec")
    _assert_utc(record["ts_eval_open"], "ts_eval_open")

    # reasoning_summary length
    summary = record.get("reasoning_summary", "")
    if len(summary) > REASONING_SUMMARY_MAX_CHARS:
        raise ValueError(f"reasoning_summary exceeds {REASONING_SUMMARY_MAX_CHARS} chars (got {len(summary)})")

    # horizons must be non-empty list of positive ints
    horizons = record.get("horizons")
    if not isinstance(horizons, list) or not horizons or not all(isinstance(h, int) and h > 0 for h in horizons):
        raise ValueError("horizons must be a non-empty list of positive integers")

    # No outcome fields allowed at write time — key must not be present with a non-None value
    for outcome_field in ("outcome", "outcome_ts", "outcome_return_pct", "outcome_direction_correct"):
        if outcome_field in record and record[outcome_field] is not None:
            raise ValueError(f"Outcome field {outcome_field!r} must be null at write time")

    # reasoning_full_ref must be relative (no absolute paths)
    ref = record.get("reasoning_full_ref", "")
    if ref and Path(ref).is_absolute():
        raise ValueError("reasoning_full_ref must be a relative path, not absolute")

    agent_id: str = record["agent_id"]
    rec_id: str = record["rec_id"]

    # Write full reasoning text to reasoning store — mutate a copy, not the caller's dict
    record = dict(record)
    reasoning_text: str = record.pop("reasoning_full_text", "")
    if reasoning_text:
        store_dir = _reasoning_store_root() / agent_id
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / f"{rec_id}.txt").write_text(reasoning_text, encoding="utf-8")
        rel_ref = str(Path(agent_id) / f"{rec_id}.txt")
        record["reasoning_full_ref"] = rel_ref

    # Append to JSONL under exclusive file lock
    log_path = _agents_root() / agent_id / "memory" / "recommendations.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(log_path), mode="a", flags=portalocker.LOCK_EX, encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    # Sync to SQLite read model
    sync_recommendation(record)


def load_records(
    agent_id: str,
    regime: str | None = None,
    last_n_days: int = 30,
    labeled_only: bool = True,
    signal: str | None = None,
    horizon_key: str | None = None,
    _db_path: Path | None = None,  # test injection only
) -> list[dict[str, Any]]:
    """Load recommendation records from the SQLite read model.

    labeled_only=True (default): joins with outcomes — only returns records
    with at least one labeled outcome. The labeler uses labeled_only=False.
    """
    db_path = _db_path or _DB_PATH
    init_db()

    cutoff_dt = datetime.now(UTC) - timedelta(days=last_n_days)
    cutoff_str = cutoff_dt.isoformat()

    conn = connect_db(str(db_path), check_same_thread=False, timeout=10, row_factory=sqlite3.Row)
    try:
        params: list[Any] = [agent_id, cutoff_str, SCHEMA_VERSION]
        where_clauses = [
            "r.agent_id = ?",
            "r.ts_rec >= ?",
            "r.schema_version = ?",
        ]

        if regime:
            where_clauses.append("r.regime_label = ?")
            params.append(regime)
        if signal:
            where_clauses.append("r.signal = ?")
            params.append(signal)

        if labeled_only:
            join_clause = "INNER JOIN outcomes o ON r.rec_id = o.rec_id"
            where_clauses.append("o.outcome_direction_correct IS NOT NULL")
            if horizon_key:
                where_clauses.append("o.horizon_key = ?")
                params.append(horizon_key)
        else:
            join_clause = "LEFT JOIN outcomes o ON r.rec_id = o.rec_id"
            if horizon_key:
                where_clauses.append("(o.horizon_key = ? OR o.horizon_key IS NULL)")
                params.append(horizon_key)

        sql = f"""
            SELECT DISTINCT r.*, o.outcome_direction_correct, o.outcome_return_pct,
                            o.outcome_ts, o.regime_label_at_eval, o.horizon_key, o.horizon_days
            FROM recommendations r
            {join_clause}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY r.ts_rec ASC
        """
        rows = conn.execute(sql, params).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            for blob_field in ("regime_context", "horizons"):
                if d.get(blob_field):
                    try:
                        d[blob_field] = json.loads(d[blob_field])
                    except json.JSONDecodeError:
                        pass
            results.append(d)
        return results
    finally:
        conn.close()
