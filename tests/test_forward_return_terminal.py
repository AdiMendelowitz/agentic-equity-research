"""(Adi 2026-09-29). an opinion dated on a day with no price bar (weekend, exchange holiday) can never fill. It is marked NO_BAR
in a side table, kept in opinions, and leaves the nightly scope once confirmed: at once on a weekend, on a second
UTC day's sighting for a weekday gap. The classification comes from the fetched
price index, never from a holiday table. populate_forward_returns admits only the watchlist plus held tickers, the same set record_opinion enforces.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

import aer.data.opinions as ops
from aer.data.cache import get_cache
from aer.models import OpinionRecord


def _frame():
    idx = pd.to_datetime(
        ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12", "2026-01-13"]
    )
    return pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]}, index=idx)


def _insert(ticker, day):
    ops.record_opinion(
        OpinionRecord(
            ticker=ticker,
            agent_name="fundamental",
            opinion_date=day,
            signal="BUY",
            confidence=0.6,
            conviction=0.5,
            drivers=[],
        )
    )


def _conn():
    return get_cache()._connect()


def _row(ticker, day):
    return (
        _conn()
        .execute(
            "select id, forward_return_5d from opinions where ticker = ? and opinion_date = ?",
            (ticker, day.isoformat()),
        )
        .fetchone()
    )


def _terminal():
    """Confirmed rows only, the ones that leave the scope."""
    return (
        _conn()
        .execute(
            "select opinion_id, reason from forward_return_terminal where confirmed_utc is not null order by opinion_id"
        )
        .fetchall()
    )


def _pending():
    return _conn().execute("select opinion_id from forward_return_terminal where confirmed_utc is null").fetchall()


def _age_first_sighting():
    _conn().execute("update forward_return_terminal set first_seen_utc = '2026-01-01T00:00:00+00:00'")
    _conn().commit()


def _prices(monkeypatch, calls=None):
    def fake(ticker, period, interval):
        if calls is not None:
            calls.append(ticker)
        return _frame()

    monkeypatch.setattr("aer.data.prices.get_price_history", fake)


# -- -----------------------------------------------------------------------------------------------------
def test_a_row_dated_on_a_day_with_no_bar_is_marked_terminal_and_kept(monkeypatch):
    _prices(monkeypatch)
    _insert("REAL", date(2026, 1, 5))  # Monday, a bar exists
    _insert("REAL", date(2026, 1, 10))  # Saturday, inside the frame, no bar
    ops.populate_forward_returns()
    trading_id, trading_fwd = _row("REAL", date(2026, 1, 5))
    saturday_id, saturday_fwd = _row("REAL", date(2026, 1, 10))
    assert trading_fwd == pytest.approx(0.05)
    assert saturday_fwd is None
    assert _terminal() == [(saturday_id, "NO_BAR")]
    assert _conn().execute("select count(*) from opinions where ticker = 'REAL'").fetchone() == (2,)


def test_a_terminal_row_leaves_the_scope_of_the_next_run(monkeypatch):
    _prices(monkeypatch)
    _insert("REAL", date(2026, 1, 10))
    ops.populate_forward_returns()
    calls = []
    _prices(monkeypatch, calls)
    assert ops.populate_forward_returns() == 0
    assert calls == []


def test_a_row_outside_the_fetched_window_is_not_marked_terminal(monkeypatch):
    """Before the first bar or after the last bar the price index says nothing about the date."""
    _prices(monkeypatch)
    _insert("REAL", date(2025, 12, 6))  # Saturday before the frame
    _insert("REAL", date(2026, 2, 7))  # Saturday after the frame
    ops.populate_forward_returns()
    assert _terminal() == []
    assert _row("REAL", date(2025, 12, 6))[1] is None
    assert _row("REAL", date(2026, 2, 7))[1] is None


def _holiday_frame(monkeypatch):
    idx = pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"])
    frame = pd.DataFrame({"Close": [100.0] * 6}, index=idx)
    monkeypatch.setattr("aer.data.prices.get_price_history", lambda t, period, interval: frame)


def test_a_weekday_gap_stays_in_scope_on_first_sighting_and_is_confirmed_on_a_later_day(monkeypatch):
    """A weekday with no bar between two bars is a holiday or a vendor hole; only a second day's sighting retires it."""
    _holiday_frame(monkeypatch)
    _insert("REAL", date(2026, 7, 3))  # NYSE holiday observed, a Friday
    ops.populate_forward_returns()
    assert _terminal() == []
    assert len(_pending()) == 1
    ops.populate_forward_returns()  # same UTC day: still pending
    assert _terminal() == []
    _age_first_sighting()
    ops.populate_forward_returns()
    assert [r[1] for r in _terminal()] == ["NO_BAR"]
    assert _pending() == []
    calls = []
    monkeypatch.setattr("aer.data.prices.get_price_history", lambda *a, **k: calls.append(a) or pd.DataFrame())
    assert ops.populate_forward_returns() == 0
    assert calls == []


def test_a_weekend_date_is_confirmed_on_first_sighting(monkeypatch):
    _prices(monkeypatch)
    _insert("REAL", date(2026, 1, 10))  # Saturday
    ops.populate_forward_returns()
    assert len(_terminal()) == 1 and _pending() == []


def test_a_bar_that_appears_later_fills_a_pending_row(monkeypatch):
    """A vendor hole that heals: the row was pending, never confirmed, and fills."""
    _holiday_frame(monkeypatch)
    _insert("REAL", date(2026, 7, 3))
    ops.populate_forward_returns()
    idx = pd.to_datetime(
        ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09", "2026-07-10"]
    )
    full = pd.DataFrame({"Close": [100, 100, 100, 101, 102, 103, 104, 110.0]}, index=idx)
    monkeypatch.setattr("aer.data.prices.get_price_history", lambda t, period, interval: full)
    ops.populate_forward_returns()
    assert _row("REAL", date(2026, 7, 3))[1] == pytest.approx(0.10)
    assert _terminal() == []


def test_a_failed_price_fetch_never_marks_a_row_terminal(monkeypatch):
    monkeypatch.setattr("aer.data.prices.get_price_history", lambda t, period, interval: pd.DataFrame())
    _insert("REAL", date(2026, 1, 10))
    ops.populate_forward_returns()
    assert _terminal() == []


# -- -----------------------------------------------------------------------------------------------------
@pytest.fixture
def scope(monkeypatch, tmp_path):
    state = tmp_path / "portfolio_state.json"
    state.write_text(json.dumps({"positions": [{"ticker": "HELD"}]}), encoding="utf-8")
    monkeypatch.setattr(ops, "_HELD_STATE_PATH", state)
    monkeypatch.setattr(ops, "_load_watchlist", lambda: ["AAPL", "MSFT"])


def test_populate_admits_only_the_watchlist_and_held_tickers(monkeypatch, scope):
    for t in ("AAPL", "HELD", "SPY", "TEST"):
        _insert(t, date(2026, 1, 5))  # the suite runs with the writer guard off
    monkeypatch.setattr(ops, "OPINION_TICKER_GUARD_ENABLED", True)
    calls = []
    _prices(monkeypatch, calls)
    ops.populate_forward_returns()
    assert sorted(calls) == ["AAPL", "HELD"]
    assert _row("SPY", date(2026, 1, 5))[1] is None
    assert _row("TEST", date(2026, 1, 5))[1] is None
    assert _row("AAPL", date(2026, 1, 5))[1] == pytest.approx(0.05)
    admitted = {r[0] for r in _conn().execute("select ticker from forward_return_admission")}
    assert admitted == {"AAPL", "HELD"}
    assert _conn().execute("select count(*) from opinions where ticker in ('SPY', 'TEST')").fetchone() == (2,)


def test_populate_admits_every_ticker_when_the_watchlist_is_unreadable(monkeypatch):
    def broken():
        raise OSError("watchlist unreadable")

    monkeypatch.setattr(ops, "_load_watchlist", broken)
    monkeypatch.setattr(ops, "OPINION_TICKER_GUARD_ENABLED", True)
    for t in ("AAPL", "SPY"):
        _insert(t, date(2026, 1, 5))
    calls = []
    _prices(monkeypatch, calls)
    ops.populate_forward_returns()
    assert sorted(calls) == ["AAPL", "SPY"]


def test_a_run_whose_only_rows_are_outside_the_scope_is_a_no_op(monkeypatch, scope):
    _insert("SPY", date(2026, 1, 5))
    monkeypatch.setattr(ops, "OPINION_TICKER_GUARD_ENABLED", True)
    calls = []
    _prices(monkeypatch, calls)
    assert ops.populate_forward_returns() == 0
    assert calls == []


def test_the_removed_ticker_grace_constant():
    from aer.utils.opinion_constants import FORWARD_RETURN_REMOVED_TICKER_GRACE_DAYS

    assert FORWARD_RETURN_REMOVED_TICKER_GRACE_DAYS == 30


def test_a_removed_ticker_is_kept_inside_the_grace_window_and_dropped_one_day_past_it(monkeypatch, scope):
    from datetime import timedelta

    from aer.utils.opinion_constants import FORWARD_RETURN_REMOVED_TICKER_GRACE_DAYS as grace

    edge = date.today() - timedelta(days=grace)
    _insert("KEEP", edge)  # newest NULL row exactly at the cutoff date: kept
    _insert("GONE", edge - timedelta(days=1))  # one day older: dropped
    monkeypatch.setattr(ops, "OPINION_TICKER_GUARD_ENABLED", True)
    calls = []
    _prices(monkeypatch, calls)
    ops.populate_forward_returns()
    assert calls == ["KEEP"]
