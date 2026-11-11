"""market-aware TTL resolver and market-hours predicate for the prefetch layer.

Provides the single source of truth for:
- is_market_hours(now=None): NYSE weekday 09:30-16:00 ET predicate (now-injectable)
- resolve_market_aware_ttl(base_ttl, *, market_aware, _now=None): routes market_aware specs
 through market_aware_ttl (parity with brief_cache:194 in src/utils/cache.py)
- resolve_prices_ttl_for_domain(domain, *, market_hours): short TTL for prices/quote during
 market hours; falls back to standard TTL constants when outside market hours
- effective_ttl(spec, *, now=None): full TTL resolver used by the driver — market-hours
 short TTL for prices/quote, else market_aware_ttl for market_aware specs, else spec.ttl
- should_repull_same_day(spec, *, now=None): True for the price-family specs in
 _INTRADAY_REPULL_SPEC_NAMES (display prices/quote + vintage prices:ohlcv_raw) in market hours
- should_repull_prices(*, written_date, market_hours): lower-level helper used by tests
 (written_date is an ISO date string; re-pull is True for a same-day write + market hours)

All time defaults are computed inside functions — never at import time (CLAUDE.md).
No direct aer.data.cache import: uses aer.utils.cache facade throughout.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

_logger = logging.getLogger(__name__)

# NYSE timezone — same constant used in simulator/daily_runner.py
_NY = ZoneInfo("America/New_York")

# Cache key prefix substrings that identify the freshness-critical prices/quote specs
# (keyed by SourceSpec.name in MANIFEST).
_PRICES_SPEC_NAME: str = "prices"
_QUOTE_SPEC_NAME: str = "prices:current"
# Raw OHLCV spec that feeds the append-only vintage modeling store.
_OHLCV_RAW_SPEC_NAME: str = "prices:ohlcv_raw"

# intraday re-pull set: freshness-critical price-family specs whose cached
# frame must be refreshed intraday so BOTH the display path (prices, prices:current)
# AND the vintage modeling path (prices:ohlcv_raw) are point-in-time-fresh on an
# intraday run.  prices:analyst_consensus is intentionally excluded — analyst targets
# are not intraday-volatile, so its once-per-day point-in-time write is sufficient and
# re-pulling it on every market-hours run would waste the fetch budget.
_INTRADAY_REPULL_SPEC_NAMES: frozenset[str] = frozenset({_PRICES_SPEC_NAME, _QUOTE_SPEC_NAME, _OHLCV_RAW_SPEC_NAME})


def is_market_hours(now: datetime | None = None) -> bool:
    """Return True during NYSE market hours (weekdays 09:30-16:00 ET).

    `now` is injectable for tests; production callers omit it and receive
    the real current time — computed here, never at module import.

    Matches _is_market_hours() in simulator/daily_runner.py exactly;
    daily_runner delegates to this function so there is one ET predicate.
    """
    _now = now if now is not None else datetime.now(_NY)
    if _now.tzinfo is None:
        _now = _now.replace(tzinfo=_NY)
    else:
        _now = _now.astimezone(_NY)

    if _now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    minutes = _now.hour * 60 + _now.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


def resolve_market_aware_ttl(
    base_ttl: int,
    *,
    market_aware: bool,
    _now: datetime | None = None,
) -> int:
    """Route `base_ttl` through market_aware_ttl when `market_aware` is True.

    Parity with brief_cache at src/utils/cache.py:194 which wraps TTL_BRIEF
    through market_aware_ttl; data/agent specs lacked this wrapper before.

    When `market_aware` is False, returns `base_ttl` unchanged (additive —
    non-market-aware specs are untouched).

    `_now` is injectable for tests (passed to market_aware_ttl's _now param).
    """
    if not market_aware:
        return base_ttl
    from aer.utils.cache import market_aware_ttl  # noqa: PLC0415 — facade import (not aer.data.cache)

    return market_aware_ttl(base_ttl, _now=_now)


def resolve_prices_ttl_for_domain(domain: str, *, market_hours: bool) -> int:
    """Return the effective TTL for the named prices/quote domain.

    Parameters
    ----------
    domain:
        ``"prices"`` returns PREFETCH_MARKET_HOURS_TTL_PRICES during market hours
        or TTL_PRICES otherwise.
        ``"quote"`` returns PREFETCH_MARKET_HOURS_TTL_QUOTE during market hours
        or TTL_QUOTE otherwise.
        Any other value returns TTL_PRICES as a safe default.
    market_hours:
        Pass the result of is_market_hours() or True/False in tests.
    """
    from aer.utils.cache import TTL_PRICES, TTL_QUOTE  # noqa: PLC0415 — facade import
    from aer.utils.constants import (  # noqa: PLC0415
        PREFETCH_MARKET_HOURS_TTL_PRICES,
        PREFETCH_MARKET_HOURS_TTL_QUOTE,
    )

    if domain == "prices":
        return PREFETCH_MARKET_HOURS_TTL_PRICES if market_hours else TTL_PRICES
    if domain == "quote":
        return PREFETCH_MARKET_HOURS_TTL_QUOTE if market_hours else TTL_QUOTE
    # Fallback for any other domain — return standard prices TTL
    return TTL_PRICES


def should_repull_prices(*, written_date: str, market_hours: bool) -> bool:
    """Return True when a same-day prices/quote write should be force-refreshed intraday.

    Parameters
    ----------
    written_date:
    ISO date string of the cached entry's write date (``YYYY-MM-DD``).
    Compared to today's date to detect same-day staleness.
    market_hours:
    Pass the result of is_market_hours. Re-pull only occurs during market hours. same-day re-pull contract: prices and quotes written earlier today must be
    refreshed while the market is open so intraday moves are captured.
    """
    if not market_hours:
        return False
    from datetime import date  # noqa: PLC0415 — computed inside function (CLAUDE.md)

    today = date.today().isoformat()
    return written_date == today


def effective_ttl(
    spec: SourceSpec,  # type: ignore[name-defined]  # noqa: F821
    *,
    now: datetime | None = None,
) -> int:
    """Single source of truth for a spec's runtime effective TTL.

    Priority
    1. prices/quote spec AND is_market_hours(now) → short market-hours TTL
    (PREFETCH_MARKET_HOURS_TTL_PRICES or PREFETCH_MARKET_HOURS_TTL_QUOTE)
    2. spec.market_aware → market_aware_ttl(spec.ttl, _now=now) for weekend/holiday extension
    3. otherwise → spec.ttl unchanged (additive; non-market-aware specs untouched)

    `now` is injectable for tests; defaults to current time inside the function.
    """
    market_hours = is_market_hours(now)

    if spec.name == _PRICES_SPEC_NAME and market_hours:
        from aer.utils.constants import PREFETCH_MARKET_HOURS_TTL_PRICES  # noqa: PLC0415

        return PREFETCH_MARKET_HOURS_TTL_PRICES

    if spec.name == _QUOTE_SPEC_NAME and market_hours:
        from aer.utils.constants import PREFETCH_MARKET_HOURS_TTL_QUOTE  # noqa: PLC0415

        return PREFETCH_MARKET_HOURS_TTL_QUOTE

    if spec.market_aware:
        return resolve_market_aware_ttl(spec.ttl, market_aware=True, _now=now)

    return spec.ttl


def should_repull_same_day(
    spec: SourceSpec,  # type: ignore[name-defined]  # noqa: F821
    *,
    now: datetime | None = None,
    as_of_date: date | None = None,
) -> bool:
    """Return True when the driver should bypass the warm cache for this spec intraday.

    Applies to the freshness-critical price-family specs in _INTRADAY_REPULL_SPEC_NAMES
    (display path prices/prices:current AND the vintage modeling spec
    prices:ohlcv_raw) during market hours, AND only when ``as_of_date`` matches
    today (or when ``as_of_date`` is None, i.e. the default today-derived run). Passing
    a historical ``as_of_date`` (e.g. for idempotency tests with a fixed past date)
    suppresses re-pull even during market hours, preserving the 30a two-pass zero-call
    proof for off-hours / historical runs.

    All other specs return False — cache-first behaviour is fully preserved.
    """
    if spec.name not in _INTRADAY_REPULL_SPEC_NAMES:
        return False
    if not is_market_hours(now):
        return False
    if as_of_date is not None:
        if as_of_date != date.today():
            return False
    return True
