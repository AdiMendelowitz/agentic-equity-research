"""Evaluation scaffolding tests."""

import sqlite3
import time
from unittest.mock import patch


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
            (ticker, f"2026-01-{(i % 28) + 1:02d}", "BUY", 0.7, "RESOLVED", 1, 1, 0, now, now),
        )
    conn.commit()
    conn.close()
    result = ws.compute_rolling_metrics()
    assert result is not None
    assert "AAPL" in result and "GOOG" in result


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
            ("AAPL", f"2024-01-{(i % 28) + 1:02d}", "BUY", 0.7, "RESOLVED", 0, 0, 0, now, now),
        )
    conn.commit()
    conn.close()
    flags = ws.should_trigger_review()
    assert "both_degraded" in flags


def test_trading_calendar_bday_consistency():
    from datetime import date

    from aer.utils.trading_calendar import add_trading_days

    start = date(2026, 5, 11)
    target = add_trading_days(start, 5)
    back = add_trading_days(target, -5)
    assert back == start
