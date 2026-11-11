"""Tests for market-aware TTL routing on data/agent SourceSpec domains.

Covers:
- src/aer/prefetch/market_aware.py: resolve_market_aware_ttl, is_prices_domain_during_market_hours
- The prefetch driver routes data/agent SourceSpec TTL through market_aware_ttl when
 SourceSpec.market_aware is True.
- A same-day re-pull for prices occurs when _is_market_hours returns True.

Uses real_key_cache (real fetcher bodies, not mocks).
"""

from __future__ import annotations

from datetime import UTC, datetime

from aer.data.cache import TTL_PRICES
from aer.utils.constants import (
    PREFETCH_MARKET_HOURS_TTL_PRICES,
    PREFETCH_MARKET_HOURS_TTL_QUOTE,
)


def test_data_domain_ttl_routes_through_market_aware(real_key_cache) -> None:
    """A market_aware SourceSpec's effective TTL equals market_aware_ttl(base) on a weekend.

    Imports the TTL resolver.
    """
    from aer.prefetch.market_aware import resolve_market_aware_ttl  # noqa: PLC0415

    # Simulate a Sunday so market_aware_ttl extends the TTL.
    sunday = datetime(2026, 6, 7, 12, 0, tzinfo=UTC)  # 2026-06-07 is a Sunday
    effective = resolve_market_aware_ttl(TTL_PRICES, market_aware=True, _now=sunday)

    from aer.data.cache import market_aware_ttl  # noqa: PLC0415

    expected = market_aware_ttl(TTL_PRICES, _now=sunday)
    assert effective == expected, (
        f"market_aware SourceSpec TTL on weekend must equal market_aware_ttl({TTL_PRICES}) "
        f"= {expected}; got {effective}"
    )


def test_prices_quote_short_ttl_in_market_hours(real_key_cache) -> None:
    """When _is_market_hours is True, prices/quote use PREFETCH_MARKET_HOURS_TTL_*.

    Imports the TTL resolver.
    """
    from aer.prefetch.market_aware import resolve_prices_ttl_for_domain  # noqa: PLC0415

    prices_ttl = resolve_prices_ttl_for_domain("prices", market_hours=True)
    quote_ttl = resolve_prices_ttl_for_domain("quote", market_hours=True)

    assert (
        prices_ttl == PREFETCH_MARKET_HOURS_TTL_PRICES
    ), f"prices TTL during market hours must be {PREFETCH_MARKET_HOURS_TTL_PRICES}; got {prices_ttl}"
    assert (
        quote_ttl == PREFETCH_MARKET_HOURS_TTL_QUOTE
    ), f"quote TTL during market hours must be {PREFETCH_MARKET_HOURS_TTL_QUOTE}; got {quote_ttl}"


def test_same_day_repull_prices_during_market_hours(real_key_cache) -> None:
    """A same-day re-pull occurs for prices when market hours are active.

    When _is_market_hours returns True, prices keys written earlier in the same session
    are treated as stale and the driver schedules a re-fetch. Imports the re-pull
    trigger.
    """
    # With market_hours=True and a same-day write timestamp, re-pull must be True.
    from datetime import date

    from aer.prefetch.market_aware import should_repull_prices  # noqa: PLC0415

    today = date.today().isoformat()
    repull = should_repull_prices(written_date=today, market_hours=True)
    assert repull is True, "same-day prices during market hours must trigger re-pull"

    # Outside market hours, no re-pull is needed.
    repull_off = should_repull_prices(written_date=today, market_hours=False)
    assert repull_off is False, "same-day prices outside market hours must NOT trigger re-pull"


def test_same_day_repull_includes_vintage_ohlcv_spec() -> None:
    """The append-only vintage OHLCV spec must re-pull intraday.

    should_repull_same_day previously gated only on 'prices'/'prices:current', so the
    vintage modeling spec 'prices:ohlcv_raw' replayed the stale cached frame on an
    intraday run - defeating the point-in-time guarantee. It must now re-pull during
    market hours, while the non-intraday-volatile 'prices:analyst_consensus' stays
    cache-first (its once-per-day point-in-time write is sufficient).
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from aer.prefetch.manifest import MANIFEST  # noqa: PLC0415
    from aer.prefetch.market_aware import should_repull_same_day  # noqa: PLC0415

    by_name = {s.name: s for s in MANIFEST}
    # A fixed weekday inside NYSE hours (Tuesday 2026-06-09, 12:00 ET).
    market_hours_now = datetime(2026, 6, 9, 12, 0, tzinfo=ZoneInfo("America/New_York"))

    ohlcv_raw = by_name["prices:ohlcv_raw"]
    assert should_repull_same_day(ohlcv_raw, now=market_hours_now, as_of_date=None) is True, (
        "vintage spec prices:ohlcv_raw must re-pull intraday so the vintage snapshot " "is point-in-time-fresh"
    )

    consensus = by_name["prices:analyst_consensus"]
    assert (
        should_repull_same_day(consensus, now=market_hours_now, as_of_date=None) is False
    ), "prices:analyst_consensus is not intraday-volatile and must stay cache-first"

    # Off-hours: even the vintage OHLCV spec must NOT re-pull (idempotency preserved).
    off_hours = datetime(2026, 6, 9, 3, 0, tzinfo=ZoneInfo("America/New_York"))
    assert (
        should_repull_same_day(ohlcv_raw, now=off_hours, as_of_date=None) is False
    ), "no spec may re-pull outside market hours"
