"""Walk-Forward Optimization (WFO) constants."""

from typing import Final

# WFO fold structure (5 folds per brief)
WFO_N_FOLDS: Final[int] = 5

# Gap between out-of-sample end and next in-sample start (prevents lookahead bias)
WFO_PURGE_GAP_BARS: Final[int] = 10

# Out-of-sample window size for each fold (55 trading days ≈ 11 weeks)
WFO_OOS_SIZE_BARS: Final[int] = 55

# Minimum in-sample bars per fold (252 trading days = 1 year)
WFO_MIN_IS_BARS: Final[int] = 252

# Computed: minimum total bars required for WFO (with 5 folds)
# TOTAL_MIN_BARS = WFO_MIN_IS_BARS + WFO_N_FOLDS * (WFO_PURGE_GAP_BARS + WFO_OOS_SIZE_BARS)
#                = 252 + 5 * (10 + 55)
#                = 252 + 5 * 65
#                = 252 + 325
#                = 577
TOTAL_MIN_BARS: Final[int] = WFO_MIN_IS_BARS + WFO_N_FOLDS * (WFO_PURGE_GAP_BARS + WFO_OOS_SIZE_BARS)
