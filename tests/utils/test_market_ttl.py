"""Tests for market_aware_ttl: weekend and NYSE holiday extension."""

from datetime import UTC, datetime

from aer.data.cache import TTL_ANALYSIS, market_aware_ttl

_UTC = UTC

# A known NYSE holiday: 2026-01-01 (New Year's Day, Thursday)
# With 1h TTL, expiry is still 2026-01-01 — a holiday.
_HOLIDAY_START = datetime(2026, 1, 1, 10, 0, tzinfo=_UTC)

# A normal Tuesday; 12h TTL expires Wednesday — trading day.
_TUESDAY = datetime(2026, 1, 6, 12, 0, tzinfo=_UTC)

# Friday 8 PM UTC; 12h TTL expires Saturday 8 AM — weekend.
_FRIDAY_EVE = datetime(2026, 1, 9, 20, 0, tzinfo=_UTC)


class TestMarketAwareTtl:
    def test_weekday_expiry_unchanged(self) -> None:
        """Expiry landing on a normal Wednesday returns base_ttl unchanged."""
        result = market_aware_ttl(TTL_ANALYSIS, _now=_TUESDAY)
        assert result == TTL_ANALYSIS

    def test_saturday_expiry_extended_to_monday(self) -> None:
        """Expiry on Saturday is pushed to Monday 14:30 UTC."""
        result = market_aware_ttl(TTL_ANALYSIS, _now=_FRIDAY_EVE)

        # Monday 2026-01-12 14:30 UTC − Friday 8 PM UTC
        monday_open = datetime(2026, 1, 12, 14, 30, tzinfo=_UTC)
        expected = int((monday_open - _FRIDAY_EVE).total_seconds())
        assert result == expected

    def test_saturday_expiry_greater_than_base(self) -> None:
        """Extended TTL exceeds base when expiry falls on a weekend."""
        result = market_aware_ttl(TTL_ANALYSIS, _now=_FRIDAY_EVE)
        assert result > TTL_ANALYSIS

    def test_holiday_expiry_extended(self) -> None:
        """Expiry on a NYSE holiday is extended to the next trading day."""
        result = market_aware_ttl(3600, _now=_HOLIDAY_START)

        # Jan 1 (holiday/Thu) → next trading day is Jan 2 (Fri)
        next_open = datetime(2026, 1, 2, 14, 30, tzinfo=_UTC)
        expected = int((next_open - _HOLIDAY_START).total_seconds())
        assert result == expected

    def test_returns_int(self) -> None:
        result = market_aware_ttl(3600, _now=_TUESDAY)
        assert isinstance(result, int)

    def test_minimum_is_base_ttl(self) -> None:
        """Result is never less than base_ttl."""
        for base in (3600, TTL_ANALYSIS, 86400):
            assert market_aware_ttl(base, _now=_TUESDAY) >= base
