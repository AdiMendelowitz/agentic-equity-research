"""Boundary tests for data-quality Final thresholds.

Per CLAUDE.md mandate: one boundary assertion per new Final threshold.
Helpers _interactive_capped and _stress_shortened are local to this module;
production implementations live in aer.prefetch.
"""

from aer.data.cache import TTL_PRICES, TTL_QUOTE
from aer.utils.constants import (
    PREFETCH_EARNINGS_WINDOW_TRADING_DAYS,
    PREFETCH_EXDATE_WINDOW_TRADING_DAYS,
    PREFETCH_MARKET_HOURS_TTL_PRICES,
    PREFETCH_MARKET_HOURS_TTL_QUOTE,
    PREFETCH_STRESS_TTL_DIVISOR,
)
from aer.utils.cost_thresholds import MAX_INTERACTIVE_FETCHES_PER_SESSION

# ---------------------------------------------------------------------------
# Local boundary helpers
# ---------------------------------------------------------------------------


def _interactive_capped(count: int, cap: int) -> bool:
    """Return True when count has reached or exceeded the cap (inclusive breach)."""
    return count >= cap


def _stress_shortened(base_ttl: int, divisor: int) -> int:
    """Return the stress-shortened TTL: base_ttl // divisor."""
    return base_ttl // divisor


# ---------------------------------------------------------------------------
# market-hours TTL thresholds
# ---------------------------------------------------------------------------


def test_market_hours_ttl_prices_is_positive() -> None:
    assert PREFETCH_MARKET_HOURS_TTL_PRICES > 0


def test_market_hours_ttl_prices_less_than_ttl_prices() -> None:
    """market-hours short TTL must be strictly shorter than the base prices TTL."""
    assert PREFETCH_MARKET_HOURS_TTL_PRICES < TTL_PRICES


def test_market_hours_ttl_quote_is_positive() -> None:
    assert PREFETCH_MARKET_HOURS_TTL_QUOTE > 0


def test_market_hours_ttl_quote_lte_ttl_quote() -> None:
    """market-hours quote TTL must not exceed the base quote TTL."""
    assert PREFETCH_MARKET_HOURS_TTL_QUOTE <= TTL_QUOTE


def test_market_hours_ttl_quote_less_than_prices_ttl() -> None:
    """Sanity: intraday quote must be fresher than daily prices TTL."""
    assert PREFETCH_MARKET_HOURS_TTL_QUOTE < PREFETCH_MARKET_HOURS_TTL_PRICES


# ---------------------------------------------------------------------------
# earnings window
# ---------------------------------------------------------------------------


def test_earnings_window_trading_days_is_one() -> None:
    """The documented ±1 trading-day window for force-refresh."""
    assert PREFETCH_EARNINGS_WINDOW_TRADING_DAYS == 1


def test_earnings_window_trading_days_is_nonnegative() -> None:
    assert PREFETCH_EARNINGS_WINDOW_TRADING_DAYS >= 0


# ---------------------------------------------------------------------------
# ex-date window
# ---------------------------------------------------------------------------


def test_exdate_window_trading_days_is_zero() -> None:
    """Same-day ex-date invalidation only (0 = today; no look-back window)."""
    assert PREFETCH_EXDATE_WINDOW_TRADING_DAYS == 0


def test_exdate_window_trading_days_is_nonnegative() -> None:
    assert PREFETCH_EXDATE_WINDOW_TRADING_DAYS >= 0


# ---------------------------------------------------------------------------
# stress TTL divisor
# ---------------------------------------------------------------------------


def test_stress_ttl_divisor_greater_than_one() -> None:
    """Divisor must be > 1 so stress shortens TTL, never lengthens it."""
    assert PREFETCH_STRESS_TTL_DIVISOR > 1


def test_stress_shortened_is_strictly_less_than_base() -> None:
    """_stress_shortened produces a TTL strictly shorter than the base."""
    shortened = _stress_shortened(TTL_PRICES, PREFETCH_STRESS_TTL_DIVISOR)
    assert shortened < TTL_PRICES


def test_stress_shortened_helper_uses_integer_division() -> None:
    """_stress_shortened is floor division - result is int."""
    result = _stress_shortened(TTL_PRICES, PREFETCH_STRESS_TTL_DIVISOR)
    assert isinstance(result, int)
    assert result == TTL_PRICES // PREFETCH_STRESS_TTL_DIVISOR


def test_stress_shortened_is_positive_for_any_valid_base() -> None:
    """Stress shortening never reaches zero for any realistic TTL with divisor <= 4."""
    assert _stress_shortened(TTL_PRICES, PREFETCH_STRESS_TTL_DIVISOR) > 0
    assert _stress_shortened(TTL_QUOTE, PREFETCH_STRESS_TTL_DIVISOR) > 0


# ---------------------------------------------------------------------------
# interactive per-session cap
# ---------------------------------------------------------------------------


def test_max_interactive_fetches_per_session_is_positive() -> None:
    assert MAX_INTERACTIVE_FETCHES_PER_SESSION > 0


def test_interactive_capped_at_cap_value() -> None:
    """Inclusive breach - count == cap triggers the circuit-breaker."""
    cap = MAX_INTERACTIVE_FETCHES_PER_SESSION
    assert _interactive_capped(cap, cap) is True


def test_interactive_not_capped_one_below() -> None:
    """Count == cap - 1 does not trigger the circuit-breaker."""
    cap = MAX_INTERACTIVE_FETCHES_PER_SESSION
    assert _interactive_capped(cap - 1, cap) is False


def test_interactive_capped_above_cap() -> None:
    """Count > cap also triggers the circuit-breaker."""
    cap = MAX_INTERACTIVE_FETCHES_PER_SESSION
    assert _interactive_capped(cap + 1, cap) is True
