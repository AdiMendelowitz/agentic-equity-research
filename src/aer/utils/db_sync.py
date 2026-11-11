"""SQLite read model for agent feedback logs.

JSONL files are the source of truth. This module maintains a query layer
that is rebuilt from JSONL on demand. Never use this as ground truth.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from aer.utils.db import connect_db

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DB_PATH = _PROJECT_ROOT / "agents" / "db" / "recommendations.db"

_initialized = False


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return connect_db(str(_DB_PATH), check_same_thread=False, timeout=10, row_factory=sqlite3.Row)


def init_db() -> None:
    """Create schema if it does not exist. Idempotent."""
    global _initialized
    if _initialized:
        return
    conn = _get_conn()
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    rec_id          TEXT NOT NULL,
                    agent_id        TEXT NOT NULL,
                    agent_version   TEXT,
                    ts_rec          TEXT NOT NULL,
                    ts_eval_open    TEXT NOT NULL,
                    ticker          TEXT NOT NULL,
                    signal          TEXT NOT NULL,
                    confidence      REAL,
                    reasoning_summary TEXT,
                    reasoning_full_ref TEXT,
                    regime_label    TEXT,
                    regime_context  TEXT,   -- JSON blob
                    horizons        TEXT,   -- JSON blob: list[int]
                    schema_version  INTEGER NOT NULL,
                    PRIMARY KEY (rec_id)
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    rec_id                  TEXT NOT NULL,
                    horizon_key             TEXT NOT NULL,
                    agent_id                TEXT NOT NULL,
                    horizon_days            INTEGER,
                    outcome_ts              TEXT,
                    outcome_return_pct      REAL,
                    outcome_direction_correct INTEGER,  -- 0/1/NULL
                    regime_label_at_eval    TEXT,
                    prices_used             TEXT,       -- JSON blob
                    PRIMARY KEY (rec_id, horizon_key)
                );
                CREATE INDEX IF NOT EXISTS idx_rec_agent_ts ON recommendations(agent_id, ts_rec);
                CREATE INDEX IF NOT EXISTS idx_out_rec ON outcomes(rec_id);
            """)
    finally:
        conn.close()
    _initialized = True


def sync_recommendation(record: dict[str, Any]) -> None:
    """Insert or replace one recommendation record into the read model."""
    conn = _get_conn()
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO recommendations
                  (rec_id, agent_id, agent_version, ts_rec, ts_eval_open, ticker,
                   signal, confidence, reasoning_summary, reasoning_full_ref,
                   regime_label, regime_context, horizons, schema_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record["rec_id"],
                    record["agent_id"],
                    record.get("agent_version"),
                    record["ts_rec"],
                    record["ts_eval_open"],
                    record["ticker"],
                    record["signal"],
                    record.get("confidence"),
                    record.get("reasoning_summary"),
                    record.get("reasoning_full_ref"),
                    record.get("regime_label"),
                    json.dumps(record.get("regime_context")),
                    json.dumps(record.get("horizons")),
                    record["schema_version"],
                ),
            )
    finally:
        conn.close()


def sync_outcome(outcome: dict[str, Any], agent_id: str) -> None:
    """Insert or replace one outcome record into the read model."""
    conn = _get_conn()
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO outcomes
                  (rec_id, horizon_key, agent_id, horizon_days, outcome_ts,
                   outcome_return_pct, outcome_direction_correct,
                   regime_label_at_eval, prices_used)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    outcome["rec_id"],
                    outcome["horizon_key"],
                    agent_id,
                    outcome.get("horizon_days"),
                    outcome.get("outcome_ts"),
                    outcome.get("outcome_return_pct"),
                    (
                        int(outcome["outcome_direction_correct"])
                        if outcome.get("outcome_direction_correct") is not None
                        else None
                    ),
                    outcome.get("regime_label_at_eval"),
                    json.dumps(outcome.get("prices_used")),
                ),
            )
    finally:
        conn.close()


def rebuild_db() -> None:
    """Drop and rebuild the read model from all JSONL files. Recovery tool."""
    global _initialized
    if _DB_PATH.exists():
        _DB_PATH.unlink()
    _initialized = False
    init_db()
    agents_root = _PROJECT_ROOT / "agents"
    if not agents_root.exists():
        return
    for rec_file in agents_root.glob("*/memory/recommendations.jsonl"):
        for line in rec_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping bad line in {rec_file}: {exc}")
                continue
            sync_recommendation(record)
    for out_file in agents_root.glob("*/memory/outcomes.jsonl"):
        agent_id = out_file.parts[-3]
        for line in out_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping bad line in {out_file}: {exc}")
                continue
            sync_outcome(record, agent_id)


init_db()
