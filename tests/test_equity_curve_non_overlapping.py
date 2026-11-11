"""(ruled, method: non-overlapping windows). get_equity_curve compounded every
opinion's 5-day forward return, so opinions on consecutive days compounded overlapping windows:
ten daily opinions of +1% each produced 1.01**10 although at most two disjoint 5-day holds fit
inside that span. Now an opinion is compounded only when its date is at least
HOLDING_PERIOD_DAYS NYSE sessions after the previously compounded one."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

import aer.data.opinions as ops
from aer.utils.opinion_constants import HOLDING_PERIOD_DAYS
from aer.utils.trading_calendar import add_trading_days


def _opinions(days, r=0.01):
    return [SimpleNamespace(opinion_date=d, forward_return_5d=r) for d in days]


def test_consecutive_daily_opinions_do_not_compound_overlapping_windows(monkeypatch):
    start = date(2026, 6, 1)
    days = [add_trading_days(start, k) for k in range(10)]
    monkeypatch.setattr(ops, "get_opinions", lambda **k: list(reversed(_opinions(days))))
    curve = ops.get_equity_curve("fundamental", "ACME")
    assert curve["dates"] == [days[0].isoformat(), days[HOLDING_PERIOD_DAYS].isoformat()]
    assert curve["cumulative_returns"] == pytest.approx([1.01, 1.01**2])


def test_window_counts_nyse_sessions_across_a_holiday(monkeypatch):
    # 2026-06-29 (Mon) + 5 sessions crosses the 2026-07-03 holiday: 06-30 07-01 07-02 07-06 07-07.
    days = [date(2026, 6, 29), date(2026, 7, 6), date(2026, 7, 7)]
    monkeypatch.setattr(ops, "get_opinions", lambda **k: _opinions(days, 0.02))
    curve = ops.get_equity_curve("fundamental", "ACME")
    assert curve["dates"] == ["2026-06-29", "2026-07-07"]


def test_null_returns_still_excluded(monkeypatch):
    rows = _opinions([date(2026, 6, 1)]) + [SimpleNamespace(opinion_date=date(2026, 6, 8), forward_return_5d=None)]
    monkeypatch.setattr(ops, "get_opinions", lambda **k: rows)
    assert ops.get_equity_curve("a", "B")["dates"] == ["2026-06-01"]
