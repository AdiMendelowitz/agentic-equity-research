"""Data quality gate — must pass before agent execution and before outcome labeling.

BAR_RESOLUTION_MINUTES=5, SESSION_BAR_COUNT=78 define the completeness denominator
(RTH 9:30–16:00 at 5-min resolution = 78 bars).
DATA_MIN_BAR_COMPLETENESS=0.98 → minimum 76.4 bars must be present.

Data failures are never misattributed to agent failures.
Always logged with source=data_quality.
"""

import logging
from dataclasses import dataclass

from aer.utils.constants import (
    BAR_RESOLUTION_MINUTES,  # noqa: F401 — imported for documentation/callers
    DATA_MAX_STALENESS_MINUTES,
    DATA_MIN_BAR_COMPLETENESS,
    SESSION_BAR_COUNT,
)

_logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when data quality checks fail before agent execution."""


@dataclass
class DataFeed:
    """Minimal data feed descriptor for quality checks."""

    staleness_minutes: float
    bar_count: int  # actual bars present
    source: str = "unknown"

    @property
    def bar_completeness(self) -> float:
        return self.bar_count / SESSION_BAR_COUNT


def assert_data_quality(feed: DataFeed) -> None:
    """Assert data quality before agent execution. Raises DataQualityError on failure.

    Failures abort the pipeline. Never misattributed to agent failures.
    """
    if feed.staleness_minutes > DATA_MAX_STALENESS_MINUTES:
        msg = (
            f"Feed stale: {feed.staleness_minutes:.1f}min > " f"{DATA_MAX_STALENESS_MINUTES}min (source={feed.source})"
        )
        _logger.error("source=data_quality %s", msg)
        raise DataQualityError(msg)
    if feed.bar_completeness < DATA_MIN_BAR_COMPLETENESS:
        msg = (
            f"Incomplete bars: {feed.bar_count}/{SESSION_BAR_COUNT} "
            f"({feed.bar_completeness:.1%} < {DATA_MIN_BAR_COMPLETENESS:.0%}) "
            f"(source={feed.source})"
        )
        _logger.error("source=data_quality %s", msg)
        raise DataQualityError(msg)


def assert_data_quality_at_label_time(feed: DataFeed) -> bool:
    """Assert data quality before outcome labeling. Returns False (does not raise) on failure.

    Failure logs the issue and signals the caller to skip the labeling batch.
    Data failures at label time are logged with source=data_quality_label.
    """
    try:
        if feed.staleness_minutes > DATA_MAX_STALENESS_MINUTES:
            _logger.error(
                "source=data_quality_label Feed stale: %.1fmin > %dmin (source=%s)",
                feed.staleness_minutes,
                DATA_MAX_STALENESS_MINUTES,
                feed.source,
            )
            return False
        if feed.bar_completeness < DATA_MIN_BAR_COMPLETENESS:
            _logger.error(
                "source=data_quality_label Incomplete bars: %d/%d (source=%s)",
                feed.bar_count,
                SESSION_BAR_COUNT,
                feed.source,
            )
            return False
        return True
    except Exception as exc:
        _logger.error("source=data_quality_label Unexpected error: %s", exc)
        return False
