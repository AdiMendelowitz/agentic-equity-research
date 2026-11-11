"""Tests for data_quality gates."""

import pytest

from aer.utils.constants import DATA_MAX_STALENESS_MINUTES, DATA_MIN_BAR_COMPLETENESS, SESSION_BAR_COUNT
from aer.utils.data_quality import DataFeed, DataQualityError, assert_data_quality, assert_data_quality_at_label_time

# Minimum bar count that satisfies the completeness threshold: ceil(0.98 * 78) = 77
_MIN_BARS = 77
# One bar below the threshold floor (76 / 78 = 0.974 < 0.98)
_INCOMPLETE_BARS = 76


def test_passes_on_fresh_complete_feed() -> None:
    """staleness=0, bar_count=SESSION_BAR_COUNT → no exception raised."""
    feed = DataFeed(staleness_minutes=0.0, bar_count=SESSION_BAR_COUNT, source="test")
    assert_data_quality(feed)  # must not raise


def test_raises_on_stale_feed() -> None:
    """staleness > DATA_MAX_STALENESS_MINUTES → DataQualityError."""
    feed = DataFeed(staleness_minutes=DATA_MAX_STALENESS_MINUTES + 5.0, bar_count=SESSION_BAR_COUNT, source="test")
    with pytest.raises(DataQualityError, match="Feed stale"):
        assert_data_quality(feed)


def test_raises_on_incomplete_bars() -> None:
    """bar_count producing completeness < DATA_MIN_BAR_COMPLETENESS → DataQualityError."""
    feed = DataFeed(staleness_minutes=0.0, bar_count=_INCOMPLETE_BARS, source="test")
    assert feed.bar_completeness < DATA_MIN_BAR_COMPLETENESS, "Precondition: feed is incomplete"
    with pytest.raises(DataQualityError, match="Incomplete bars"):
        assert_data_quality(feed)


def test_boundary_staleness_passes() -> None:
    """staleness exactly at DATA_MAX_STALENESS_MINUTES → passes (boundary is inclusive)."""
    feed = DataFeed(staleness_minutes=float(DATA_MAX_STALENESS_MINUTES), bar_count=SESSION_BAR_COUNT, source="test")
    assert_data_quality(feed)  # must not raise at the exact boundary


def test_label_time_returns_false_not_raises() -> None:
    """Stale feed at label time → returns False, does not raise DataQualityError."""
    feed = DataFeed(staleness_minutes=DATA_MAX_STALENESS_MINUTES + 1.0, bar_count=SESSION_BAR_COUNT, source="test")
    result = assert_data_quality_at_label_time(feed)
    assert result is False


def test_label_time_returns_true_on_good_feed() -> None:
    """Fresh, complete feed at label time → returns True."""
    feed = DataFeed(staleness_minutes=0.0, bar_count=SESSION_BAR_COUNT, source="test")
    result = assert_data_quality_at_label_time(feed)
    assert result is True
