"""Evaluation scaffolding tests."""

import hashlib
import logging
import re
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

_OLD_SCHEMA = """
CREATE TABLE agent_eval (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    brief_date          TEXT    NOT NULL,
    recommendation      TEXT    NOT NULL,
    conviction          REAL    NOT NULL,
    ml_prediction_skipped INTEGER NOT NULL DEFAULT 0,
    resolution_date     TEXT,
    resolution_attempts INTEGER NOT NULL DEFAULT 0,
    resolution_status   TEXT    NOT NULL DEFAULT 'PENDING',
    created_at          REAL    NOT NULL,
    updated_at          REAL    NOT NULL
);
CREATE INDEX idx_eval_ticker_date ON agent_eval(ticker, brief_date);
"""


def _old_store(path: Path, rows: list[tuple[str, str]]) -> None:
    """A WAL store with the pre-run_id schema, so that opening it again leaves the file bytes alone."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_OLD_SCHEMA)
    conn.executemany(
        "INSERT INTO agent_eval (ticker, brief_date, recommendation, conviction, created_at, updated_at) "
        "VALUES (?,?,'BUY',0.7,1.0,1.0)",
        rows,
    )
    conn.commit()
    conn.close()


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _columns(path: Path) -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        return [r[1] for r in conn.execute("PRAGMA table_info(agent_eval)")]
    finally:
        conn.close()


def _has_unique_index(path: Path) -> bool:
    conn = sqlite3.connect(str(path))
    try:
        return any(r[2] == 1 for r in conn.execute("PRAGMA index_list(agent_eval)"))
    finally:
        conn.close()


def _index_names(path: Path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {r[1] for r in conn.execute("PRAGMA index_list(agent_eval)")}
    finally:
        conn.close()


class _ConnProxy:
    """Delegates to a real connection, but lets execute() intercept the run_id ALTER TABLE statement."""

    def __init__(self, conn: sqlite3.Connection, on_alter) -> None:
        self._conn = conn
        self._on_alter = on_alter

    def execute(self, sql, *args):
        if sql == "ALTER TABLE agent_eval ADD COLUMN run_id TEXT":
            return self._on_alter(self._conn, sql)
        return self._conn.execute(sql, *args)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _proxy_connect(monkeypatch, ws, on_alter) -> None:
    real_connect = ws._connect
    monkeypatch.setattr(ws, "_connect", lambda: _ConnProxy(real_connect(), on_alter))


def test_synth_correct_hold_is_none():
    from aer.evaluation.eval_models import compute_synth_correct

    assert compute_synth_correct("HOLD", "UP") is None
    assert compute_synth_correct("HOLD", "DOWN") is None


def test_synth_correct_buy_correct():
    from aer.evaluation.eval_models import compute_synth_correct

    assert compute_synth_correct("BUY", "UP") is True
    assert compute_synth_correct("BUY", "DOWN") is False


def test_synth_correct_avoid_is_bearish():
    from aer.evaluation.eval_models import compute_synth_correct

    assert compute_synth_correct("AVOID", "DOWN") is True
    assert compute_synth_correct("AVOID", "UP") is False


def test_synth_correct_sell_is_bearish():
    from aer.evaluation.eval_models import compute_synth_correct

    assert compute_synth_correct("SELL", "DOWN") is True
    assert compute_synth_correct("SELL", "UP") is False


def test_sources_valid_rejects_bare_strings():
    from aer.evaluation.process_checker import check_structural_quality
    from aer.models import AgentOutput, InvestmentBrief

    fundamental = AgentOutput(
        agent_name="fundamental",
        content="analysis",
        sources=["not-a-url", "also-not"],
        confidence=0.7,
    )
    brief = InvestmentBrief(
        ticker="AAPL",
        company_name="Apple",
        fundamental=fundamental,
    )
    quality = check_structural_quality(brief)
    assert quality["sources_appear_valid"] is False


def test_structural_quality_all_agents_none():
    from aer.evaluation.process_checker import check_structural_quality
    from aer.models import InvestmentBrief

    brief = InvestmentBrief(ticker="AAPL", company_name="Apple")
    quality = check_structural_quality(brief)
    assert isinstance(quality, dict)
    assert "ml_prediction_present" in quality
    assert "backtest_present" in quality
    assert quality["ml_prediction_present"] is False
    assert quality["backtest_present"] is False
    assert quality["agents_complete"] is False, "a brief with no agent output is not complete"
    assert "all_agents_agree_direction" not in quality


def test_record_brief_writes_pending_row(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws
    from aer.models import InvestmentBrief

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    brief = InvestmentBrief(ticker="AAPL", company_name="Apple", recommendation="BUY", conviction=0.75)
    quality: dict = {}
    ws.record_brief_generated(brief, quality)
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    rows = conn.execute("SELECT resolution_status FROM agent_eval").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "PENDING"


def test_same_brief_twice_on_one_day_keeps_the_first_row(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws
    from aer.models import InvestmentBrief

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    ws.record_brief_generated(
        InvestmentBrief(ticker="AAPL", company_name="A", recommendation="BUY", conviction=0.9), {}
    )
    ws.record_brief_generated(
        InvestmentBrief(ticker="AAPL", company_name="A", recommendation="SELL", conviction=0.2), {}
    )
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    rows = conn.execute("SELECT recommendation, conviction FROM agent_eval").fetchall()
    conn.close()
    assert rows == [("BUY", 0.9)]


def test_skipped_rerun_is_logged_with_a_running_count(tmp_path, monkeypatch, caplog):
    import aer.evaluation.weekly_scorer as ws
    from aer.models import InvestmentBrief

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    monkeypatch.setattr(ws, "_skipped_reruns", 0)
    brief = InvestmentBrief(ticker="AAPL", company_name="A", recommendation="BUY", conviction=0.9)
    ws.record_brief_generated(brief, {})
    with caplog.at_level(logging.WARNING, logger=ws.logger.name):
        ws.record_brief_generated(brief, {})
        ws.record_brief_generated(brief, {})
    skips = [r.getMessage() for r in caplog.records if "skipped" in r.getMessage()]
    assert len(skips) == 2
    assert "AAPL" in skips[0] and "1 skipped" in skips[0]
    assert "2 skipped" in skips[1]


def test_other_constraint_failure_still_raises(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws
    from aer.models import InvestmentBrief

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    brief = InvestmentBrief.model_construct(ticker="AAPL", company_name="A", recommendation="BUY", conviction=None)
    with pytest.raises(sqlite3.IntegrityError):
        ws.record_brief_generated(brief, {})


def test_init_db_refuses_a_store_with_duplicates_and_leaves_it_unchanged(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws

    db = tmp_path / "eval.db"
    _old_store(db, [("AAPL", "2026-01-01"), ("AAPL", "2026-01-01"), ("GOOG", "2026-01-01")])
    monkeypatch.setattr(ws, "_EVAL_DB_PATH", db)
    before = _md5(db)
    with pytest.raises(ws.DuplicateTickerDayError, match="remove_duplicate_ticker_days"):
        ws._init_db()
    assert _md5(db) == before
    assert "run_id" not in _columns(db)
    assert not _has_unique_index(db)


def test_remove_duplicates_keeps_the_lowest_id_and_unblocks_init_db(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws

    db = tmp_path / "eval.db"
    _old_store(db, [("AAPL", "2026-01-01"), ("AAPL", "2026-01-01"), ("GOOG", "2026-01-01"), ("AAPL", "2026-01-01")])
    monkeypatch.setattr(ws, "_EVAL_DB_PATH", db)
    assert ws.remove_duplicate_ticker_days() == 2
    conn = sqlite3.connect(str(db))
    ids = [r[0] for r in conn.execute("SELECT id FROM agent_eval ORDER BY id")]
    conn.close()
    assert ids == [1, 3]
    ws._init_db()
    assert _has_unique_index(db)


def test_remove_duplicates_on_missing_store_or_table_returns_zero(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws

    missing = tmp_path / "sub" / "eval.db"
    monkeypatch.setattr(ws, "_EVAL_DB_PATH", missing)
    assert ws.remove_duplicate_ticker_days() == 0
    assert not missing.exists() and not missing.parent.exists()

    other = tmp_path / "other.db"
    conn = sqlite3.connect(str(other))
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(ws, "_EVAL_DB_PATH", other)
    assert ws.remove_duplicate_ticker_days() == 0


def test_old_schema_store_migrates_and_keeps_its_rows(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws

    db = tmp_path / "eval.db"
    _old_store(db, [("AAPL", "2026-01-01"), ("AAPL", "2026-01-02")])
    assert "run_id" not in _columns(db)
    monkeypatch.setattr(ws, "_EVAL_DB_PATH", db)
    ws._init_db()
    ws._init_db()
    assert "run_id" in _columns(db)
    assert {"idx_eval_ticker_date", "uq_eval_ticker_brief_date"} <= _index_names(db)
    conn = sqlite3.connect(str(db))
    assert conn.execute("SELECT COUNT(*) FROM agent_eval").fetchone()[0] == 2
    conn.close()


def test_concurrent_migration_duplicate_column_is_tolerated(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws

    db = tmp_path / "eval.db"
    _old_store(db, [("AAPL", "2026-01-01")])
    monkeypatch.setattr(ws, "_EVAL_DB_PATH", db)

    def alter_twice(conn, sql):
        conn.execute(sql)
        return conn.execute(sql)

    _proxy_connect(monkeypatch, ws, alter_twice)
    ws._init_db()
    assert "run_id" in _columns(db)
    assert {"idx_eval_ticker_date", "uq_eval_ticker_brief_date"} <= _index_names(db)


def test_migration_failure_other_than_duplicate_column_propagates(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws

    db = tmp_path / "eval.db"
    _old_store(db, [("AAPL", "2026-01-01")])
    monkeypatch.setattr(ws, "_EVAL_DB_PATH", db)

    def locked(conn, sql):
        raise sqlite3.OperationalError("database is locked")

    _proxy_connect(monkeypatch, ws, locked)
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        ws._init_db()
    assert "run_id" not in _columns(db)


def test_half_migrated_store_with_duplicates_is_refused_without_an_index(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws

    db = tmp_path / "eval.db"
    _old_store(db, [("AAPL", "2026-01-01"), ("AAPL", "2026-01-01")])
    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE agent_eval ADD COLUMN run_id TEXT")
    conn.commit()
    conn.close()
    monkeypatch.setattr(ws, "_EVAL_DB_PATH", db)
    with pytest.raises(ws.DuplicateTickerDayError, match="1 ticker-day with more than one row"):
        ws._init_db()
    assert "uq_eval_ticker_brief_date" not in _index_names(db)


def test_inserts_in_one_process_share_a_non_empty_run_id(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws
    from aer.models import InvestmentBrief

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    for ticker in ("AAPL", "GOOG"):
        ws.record_brief_generated(
            InvestmentBrief(ticker=ticker, company_name=ticker, recommendation="BUY", conviction=0.5), {}
        )
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    ids = {r[0] for r in conn.execute("SELECT run_id FROM agent_eval")}
    conn.close()
    assert ids == {ws._RUN_ID}
    assert re.fullmatch(r"[0-9a-f]{32}", ws._RUN_ID)


def test_resolve_increments_attempts_on_failure(tmp_path, monkeypatch):

    import aer.evaluation.weekly_scorer as ws
    from aer.models import InvestmentBrief

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    brief = InvestmentBrief(ticker="FAIL", company_name="FailCo", recommendation="BUY", conviction=0.5)
    ws.record_brief_generated(brief, {})
    # Backdate resolution_date so the row is eligible for resolution
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    conn.execute("UPDATE agent_eval SET resolution_date=?", ("2020-01-01",))
    conn.commit()
    conn.close()
    with patch("aer.evaluation.weekly_scorer.yf.download", side_effect=Exception("network error")):
        ws.resolve_pending_outcomes()
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    row = conn.execute("SELECT resolution_attempts, resolution_status FROM agent_eval").fetchone()
    conn.close()
    assert row[0] == 1
    assert row[1] == "PENDING"


def test_resolve_marks_unresolvable_at_max_attempts(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws
    from aer.ml.constants import MAX_RESOLUTION_ATTEMPTS
    from aer.models import InvestmentBrief

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    brief = InvestmentBrief(ticker="FAIL", company_name="FailCo", recommendation="SELL", conviction=0.6)
    ws.record_brief_generated(brief, {})
    past_date = "2020-01-01"
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    # Set attempts to MAX-1 so the row is eligible (WHERE clause: resolution_attempts < MAX)
    # On the next failure attempt it will reach MAX and be marked UNRESOLVABLE
    conn.execute(
        "UPDATE agent_eval SET resolution_attempts=?, resolution_date=?",
        (MAX_RESOLUTION_ATTEMPTS - 1, past_date),
    )
    conn.commit()
    conn.close()
    with patch("aer.evaluation.weekly_scorer.yf.download", side_effect=Exception("fail")):
        ws.resolve_pending_outcomes()
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    row = conn.execute("SELECT resolution_status FROM agent_eval").fetchone()
    conn.close()
    assert row[0] == "UNRESOLVABLE"


def test_rolling_metrics_none_below_threshold(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    ws._init_db()
    result = ws.compute_rolling_metrics()
    assert result is None


def test_rolling_metrics_per_ticker_not_pooled(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    ws._init_db()
    now = time.time()
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    for i in range(220):
        ticker = "AAPL" if i < 110 else "GOOG"
        conn.execute(
            "INSERT INTO agent_eval (ticker, brief_date, recommendation, conviction, resolution_status, "
            "synth_correct, ml_correct, resolution_attempts, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ticker, (date(2026, 1, 1) + timedelta(days=i)).isoformat(), "BUY", 0.7, "RESOLVED", 1, 1, 0, now, now),
        )
    conn.commit()
    conn.close()
    result = ws.compute_rolling_metrics()
    assert result is not None
    assert "AAPL" in result and "GOOG" in result
    assert result["AAPL"]["unresolvable_fraction"] == 0.0


def test_rolling_metrics_count_unresolvable_rows(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    ws._init_db()
    now = time.time()
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    for i in range(120):
        status = "RESOLVED" if i < 100 else "UNRESOLVABLE"
        correct = 1 if status == "RESOLVED" else None
        conn.execute(
            "INSERT INTO agent_eval (ticker, brief_date, recommendation, conviction, resolution_status, "
            "synth_correct, ml_correct, resolution_attempts, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "AAPL",
                (date(2026, 1, 1) + timedelta(days=i)).isoformat(),
                "BUY",
                0.7,
                status,
                correct,
                correct,
                0,
                now,
                now,
            ),
        )
    conn.commit()
    conn.close()
    result = ws.compute_rolling_metrics()
    assert result is not None
    assert result["AAPL"]["n"] == 100
    assert result["AAPL"]["unresolvable_fraction"] == 20 / 120


def test_review_flags_suppressed_before_baseline(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    ws._init_db()
    flags = ws.should_trigger_review()
    assert all(v is False for v in flags.values())


def test_both_degraded_flag_set(tmp_path, monkeypatch):
    import aer.evaluation.weekly_scorer as ws
    from aer.ml.constants import BASELINE_WINDOW_WEEKS

    monkeypatch.setattr(ws, "_EVAL_DB_PATH", tmp_path / "eval.db")
    ws._init_db()
    min_n = int(BASELINE_WINDOW_WEEKS * 5) + 1
    now = time.time()
    conn = sqlite3.connect(str(tmp_path / "eval.db"))
    for i in range(min_n):
        conn.execute(
            "INSERT INTO agent_eval (ticker, brief_date, recommendation, conviction, resolution_status, "
            "synth_correct, ml_correct, resolution_attempts, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("AAPL", (date(2024, 1, 1) + timedelta(days=i)).isoformat(), "BUY", 0.7, "RESOLVED", 0, 0, 0, now, now),
        )
    conn.commit()
    conn.close()
    flags = ws.should_trigger_review()
    assert "both_degraded" in flags


def test_trading_calendar_bday_consistency():
    from aer.utils.trading_calendar import add_trading_days

    start = date(2026, 5, 11)
    target = add_trading_days(start, 5)
    back = add_trading_days(target, -5)
    assert back == start
