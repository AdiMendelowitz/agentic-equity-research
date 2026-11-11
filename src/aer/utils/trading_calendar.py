"""Trading-day arithmetic. Single source of truth for all horizon calculations.

One layer, NYSE-accurate throughout, R5, Adi 2026-09-24): every function here
is backed by ``pandas-market-calendars`` (XNYS), including ``add_trading_days`` and
``trading_days_between``, which until 2026-09-24 were holiday-blind ``pandas`` business-day helpers
with a second hand-rolled copy in ``src/prefetch/earnings_window.py``. All nine call sites
(features label horizon, weekly scorer, alt-data eval, earnings window) now get NYSE sessions.
This module is the ONLY importer of ``pandas_market_calendars`` in the repo, and
tests/test_trading_calendar.py fails if another module hand-rolls trading-day arithmetic.

All functions are pure: no wall-clock reads, no I/O. ``is_stale`` takes both timestamps as
arguments so tests can pin the clock (B2b clarifications v2 §3).
"""

from __future__ import annotations

import bisect
import functools
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

_CALENDAR_NAME = "XNYS"  # NYSE; "NYSE" is an alias of the same calendar in pandas-market-calendars
_ET = ZoneInfo("America/New_York")
_LOOKAHEAD_WINDOW_DAYS = 15  # widest US market gap (holiday + long weekend) is < 5 days; 15 is safe margin


# ── NYSE-accurate calendar (B2b) ──────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _calendar() -> mcal.MarketCalendar:
    return mcal.get_calendar(_CALENDAR_NAME)


_SESSION_SPAN = ("1990-01-01", "2045-12-31")  # one cached session list serves every horizon lookup


@functools.lru_cache(maxsize=1)
def _sessions() -> tuple[date, ...]:
    return tuple(ts.date() for ts in _calendar().valid_days(start_date=_SESSION_SPAN[0], end_date=_SESSION_SPAN[1]))


def add_trading_days(start: date, n: int) -> date:
    """The n-th NYSE session after ``start`` (n > 0) or before it (n < 0).

    n == 0 returns ``start`` when it is a session, else the next session (the roll-forward the
    old ``BDay(0)`` gave on weekends, now also across holidays).
    """
    sess = _sessions()
    if n == 0:
        i = bisect.bisect_left(sess, start)
    elif n > 0:
        i = bisect.bisect_right(sess, start) + n - 1
    else:
        i = bisect.bisect_left(sess, start) + n
    if not 0 <= i < len(sess):
        raise ValueError(f"add_trading_days({start}, {n}) falls outside {_SESSION_SPAN}")
    return sess[i]


def trading_days_between(start: date, end: date) -> int:
    """Number of NYSE sessions ``s`` with start < s <= end (negative when end < start)."""
    sess = _sessions()
    return bisect.bisect_right(sess, end) - bisect.bisect_right(sess, start)


@functools.lru_cache(maxsize=8192)
def _valid_days(start_iso: str, end_iso: str) -> tuple[date, ...]:
    idx = _calendar().valid_days(start_date=start_iso, end_date=end_iso)
    return tuple(ts.date() for ts in idx)


def is_trading_day(d: date) -> bool:
    """True if ``d`` is an NYSE trading day (weekday and not a market holiday)."""
    return d in _valid_days(d.isoformat(), d.isoformat())


def next_trading_day(d: date) -> date:
    """The first NYSE trading day strictly after ``d``."""
    start = d + timedelta(days=1)
    end = start + timedelta(days=_LOOKAHEAD_WINDOW_DAYS)
    while True:
        days = _valid_days(start.isoformat(), end.isoformat())
        if days:
            return days[0]
        start = end + timedelta(days=1)
        end = start + timedelta(days=_LOOKAHEAD_WINDOW_DAYS)


def most_recent_expected_run(now_et_date: date) -> date:
    """The most recent NYSE trading day on or before ``now_et_date``.

    DASHBOARD STALENESS ONLY (B2b corrections v2 §2). Today counts as soon as it is a trading
    day, because the pipeline runs ~01:30 ET — well before any dashboard view. This value is
    strictly earlier than that run's ``intended_entry_date`` (which is ``next_trading_day`` of
    it), which is why it must never appear in the simulator's entry gate.
    """
    d = now_et_date
    for _ in range(_LOOKAHEAD_WINDOW_DAYS + 1):
        if is_trading_day(d):
            return d
        d -= timedelta(days=1)
    # Unreachable in practice (a trading day always exists within ~4 days); explicit for safety.
    raise RuntimeError(f"No NYSE trading day found within {_LOOKAHEAD_WINDOW_DAYS} days of {now_et_date}")


def is_stale(artifact_ts_utc: datetime, now_utc: datetime, cal: object | None = None) -> bool:
    """True if a newer pipeline run should have produced a fresher artifact than this one.

    Both inputs must be timezone-aware UTC; they are converted to America/New_York before the
    day-level comparison. ``cal`` is accepted for signature compatibility and injection but the
    module's cached NYSE calendar is authoritative.

    Day-granular by design (matches ``most_recent_expected_run``'s ``date`` signature): the sole
    edge is a view between 00:00 and ~02:00 ET before the day's run lands, which a date-only
    function cannot resolve. Acceptable because the pipeline runs 01:00–02:00 ET and the
    dashboard is viewed well after.
    """
    if artifact_ts_utc.tzinfo is None or now_utc.tzinfo is None:
        raise ValueError("is_stale requires timezone-aware UTC datetimes")
    artifact_et_date = artifact_ts_utc.astimezone(_ET).date()
    now_et_date = now_utc.astimezone(_ET).date()
    return artifact_et_date < most_recent_expected_run(now_et_date)


_MARKET_CLOSE_HOUR_ET = 16  # NYSE regular-session close; a session is "complete" only after this


def last_completed_session(now_et: datetime) -> date:
    """Most recent NYSE trading day whose 16:00 ET close is <= now_et. now_et must be tz-aware.

    run_pipeline stamps data_asof_date with this. At the scheduled 01:00-02:00 ET run today's
    session hasn't opened, so this returns the prior trading day; a post-16:00 run returns today.
    """
    if now_et.tzinfo is None:
        raise ValueError("last_completed_session requires a tz-aware datetime")
    now_et = now_et.astimezone(_ET)
    cutoff = now_et.date() if now_et.hour >= _MARKET_CLOSE_HOUR_ET else now_et.date() - timedelta(days=1)
    d = cutoff
    for _ in range(_LOOKAHEAD_WINDOW_DAYS + 1):
        if is_trading_day(d):
            return d
        d -= timedelta(days=1)
    raise RuntimeError(f"No completed NYSE session found within {_LOOKAHEAD_WINDOW_DAYS} days of {now_et}")


def _utc(*args: int) -> datetime:
    """Test helper: build a tz-aware UTC datetime."""
    return datetime(*args, tzinfo=UTC)
