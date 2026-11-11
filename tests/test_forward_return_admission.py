"""R-D (Adi 2026-09-28), task 7a: forward-return admission cannot be starved, and the opinion writer admits only
tickers the system trades or watches. 117 unpriceable synthetic tickers held all 10 admission slots."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

import aer.data.opinions as ops
from aer.data.cache import get_cache
from aer.models import OpinionRecord
from aer.utils.opinion_constants import FORWARD_RETURN_MAX_FAILED_ATTEMPTS


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


def _fwd(ticker):
    row = (
        get_cache()._connect().execute("select forward_return_5d from opinions where ticker = ?", (ticker,)).fetchone()
    )
    return row[0]


def _prices(priced: set[str]):
    def fake(ticker, period, interval):
        return _frame() if ticker in priced else pd.DataFrame()

    return fake


def test_the_attempt_limit_constant():
    assert FORWARD_RETURN_MAX_FAILED_ATTEMPTS == 5


def test_the_per_run_cap_covers_the_real_backlog_in_one_night():
    """Adi 2026-09-28: cap 10 -> 60; 57 real tickers carry NULL rows after the archive."""
    from aer.utils.opinion_constants import MAX_FORWARD_RETURN_TICKERS_PER_RUN

    assert MAX_FORWARD_RETURN_TICKERS_PER_RUN == 60


def test_unpriceable_tickers_cannot_starve_a_real_one(monkeypatch):
    """Cap 2, three older unpriceable tickers, one newer real one: the real one fills within three runs.
    Oldest-first admission (the pre-fix rule) never reached it."""
    monkeypatch.setattr("aer.utils.opinion_constants.MAX_FORWARD_RETURN_TICKERS_PER_RUN", 2)
    monkeypatch.setattr("aer.data.prices.get_price_history", _prices({"REAL"}))
    for t in ("DEADA", "DEADB", "DEADC"):
        _insert(t, date(2026, 1, 2))
    _insert("REAL", date(2026, 1, 5))
    for _ in range(3):
        ops.populate_forward_returns()
    assert _fwd("REAL") == pytest.approx(0.05)


def test_a_ticker_is_retired_after_the_attempt_limit_and_its_rows_are_kept(monkeypatch):
    monkeypatch.setattr("aer.data.prices.get_price_history", _prices(set()))
    _insert("DEADA", date(2026, 1, 5))
    for _ in range(FORWARD_RETURN_MAX_FAILED_ATTEMPTS):
        ops.populate_forward_returns()
    conn = get_cache()._connect()
    status, attempts = conn.execute(
        "select status, attempts from forward_return_admission where ticker = 'DEADA'"
    ).fetchone()
    assert (status, attempts) == ("UNPRICEABLE", FORWARD_RETURN_MAX_FAILED_ATTEMPTS)
    calls = []
    monkeypatch.setattr("aer.data.prices.get_price_history", lambda *a, **k: calls.append(a) or pd.DataFrame())
    ops.populate_forward_returns()
    assert calls == []
    assert conn.execute("select count(*) from opinions where ticker = 'DEADA'").fetchone() == (1,)


def test_a_successful_fetch_resets_the_failure_count(monkeypatch):
    _insert("FLAKY", date(2026, 1, 5))
    monkeypatch.setattr("aer.data.prices.get_price_history", _prices(set()))
    ops.populate_forward_returns()
    monkeypatch.setattr("aer.data.prices.get_price_history", _prices({"FLAKY"}))
    ops.populate_forward_returns()
    conn = get_cache()._connect()
    assert conn.execute("select attempts, status from forward_return_admission where ticker='FLAKY'").fetchone() == (
        0,
        None,
    )


def test_the_fetch_window_is_sized_from_admitted_rows_only(monkeypatch):
    conn = get_cache()._connect()
    _insert("OLD", date(2024, 1, 2))
    _insert("REAL", date(2026, 1, 5))
    ops.ensure_admission_table(conn)
    conn.execute("insert into forward_return_admission(ticker, attempts, status) values ('OLD', 5, 'UNPRICEABLE')")
    conn.commit()
    periods = []

    def fake(ticker, period, interval):
        periods.append(period)
        return _frame()

    monkeypatch.setattr("aer.data.prices.get_price_history", fake)
    ops.populate_forward_returns()
    span = (date.today() - date(2026, 1, 5)).days
    from aer.utils.opinion_constants import FORWARD_RETURN_PERIOD_LADDER_BARS as ladder

    expected = next((b for b in ladder if b >= span), ladder[-1])
    assert periods == [f"{expected}d"]


# -- write guard ----------------------------------------------------------------------------------
@pytest.fixture
def guard_on(monkeypatch, tmp_path):
    state = tmp_path / "portfolio_state.json"
    state.write_text(json.dumps({"positions": [{"ticker": "HELD"}]}), encoding="utf-8")
    monkeypatch.setattr(ops, "_HELD_STATE_PATH", state)
    monkeypatch.setattr(ops, "_load_watchlist", lambda: ["AAPL", "MSFT"])
    monkeypatch.setattr(ops, "OPINION_TICKER_GUARD_ENABLED", True)


def test_the_writer_rejects_a_ticker_outside_the_watchlist_and_holdings(guard_on):
    _insert("TAB12", date(2026, 1, 5))
    n = get_cache()._connect().execute("select count(*) from opinions where ticker='TAB12'").fetchone()[0]
    assert n == 0


def test_the_writer_accepts_watchlist_and_held_tickers(guard_on):
    _insert("AAPL", date(2026, 1, 5))
    _insert("HELD", date(2026, 1, 5))
    conn = get_cache()._connect()
    assert conn.execute("select count(*) from opinions where ticker in ('AAPL','HELD')").fetchone() == (2,)


def test_the_guard_is_on_outside_the_test_session():
    """conftest switches it off for the suite; the shipped default is on."""
    import importlib.util

    spec = importlib.util.find_spec("aer.data.opinions")
    source = spec.loader.get_source("aer.data.opinions")
    assert "OPINION_TICKER_GUARD_ENABLED: bool = True" in source
