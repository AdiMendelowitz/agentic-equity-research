"""Named constants for opinion tracking system.

All thresholds, signal mappings, and cache TTLs defined here.
Zero imports from src/data or agents to prevent circular dependencies.
"""

from typing import Final

# Minimum history requirements for analytics
MIN_HISTORY_DAYS: Final[int] = 30  # Minimum history required for backfill
MIN_OPINIONS_FOR_METRICS: Final[int] = 20  # Minimum opinions to compute Sharpe
MIN_VALID_RETURNS_FOR_SHARPE: Final[int] = 20  # Minimum non-NULL forward_return_5d for Sharpe

# Hit-rate calculation thresholds
BINARY_HIT_THRESHOLD: Final[float] = 0.0001  # Forward return threshold for binary accuracy (0.01% = negligible drift)
HOLD_THRESHOLD: Final[float] = 0.01  # 1% threshold: |return| < HOLD_THRESHOLD marks HOLD as correct

# Confidence and conviction bounds
CONFIDENCE_MIN: Final[float] = 0.0
CONFIDENCE_MAX: Final[float] = 1.0
CONVICTION_MIN: Final[float] = 0.0
CONVICTION_MAX: Final[float] = 1.0

# Cache retention
TTL_OPINIONS: Final[int] = 90 * 86400  # 90-day retention for opinion cache
TTL_METRICS: Final[int] = 12 * 3600  # 12h cache TTL for hit-rate metrics

# Forward return window
HOLDING_PERIOD_DAYS: Final[int] = 5  # Forward return window

# Forward-return backfill: period ladder + per-run ticker admission bound.
# The ladder is in BARS, not calendar days. yfinance `Nd` returns N BARS: records the
# period regex ^[1-9]\d*(d|wk|mo|y)$ as verified against yfinance 1.3.0, and a 437d fetch was
# measured returning 437 rows spanning roughly 638 calendar days. N bars span about N * 7 / 5
# calendar days, so a rung chosen to cover a CALENDAR-day span over-provisions by roughly 40%.
# Over-fetch is harmless; quantizing is what keeps the cache key stable from one
# nightly run to the next, which per-row `days_back` arithmetic did not.
FORWARD_RETURN_PERIOD_LADDER_BARS: Final[tuple[int, ...]] = (30, 60, 90, 180, 365, 730)

# One price fetch per admitted ticker per run. Ruled 2026-09-28 (Adi): 10 -> 60. After the archive the
# backlog is 57 real tickers (174 before, 117 of them synthetic), so 60 clears it in one nightly run; with
# never-attempted-first admission and UNPRICEABLE retirement (R-D) no ticker can hold a slot indefinitely.
# The previous value 10 was a first-run throttle for a hand-check that the admission fix replaces.
MAX_FORWARD_RETURN_TICKERS_PER_RUN: Final[int] = 60

# R-D (Adi 2026-09-28), a ticker whose price fetch fails or returns no history this many runs in a row is
# marked UNPRICEABLE in forward_return_admission and never admitted again; its opinion rows are kept.
FORWARD_RETURN_MAX_FAILED_ATTEMPTS: Final[int] = 5

# (Adi 2026-09-29, executor default): populate_forward_returns admits the watchlist plus held positions. A
# ticker that left both stays admitted while its newest NULL row is at most this many calendar days old, so a
# recently removed name still fills its last rows. Older residue (SPY, TEST, TICK0 to TICK4) is not fetched.
FORWARD_RETURN_REMOVED_TICKER_GRACE_DAYS: Final[int] = 30

# Backfill configuration

# Signal mapping: used by opinion analytics
SIGNAL_DIRECTION_MAP: Final[dict[str, float]] = {
    "BUY": 1.0,
    "HOLD": 0.0,
    "SELL": -1.0,
    "AVOID": -1.0,  # explicit AVOID handling
}
