"""ML layer constants. All values are fixed spec - do not derive or infer."""

from pathlib import Path

# Path anchored to this file's location - survives changes of working directory
from aer.paths import PROJECT_ROOT as _ML_ROOT

MODEL_DIR: Path = _ML_ROOT / "ml_models"

# Cross-validation
PREDICTION_HORIZON_DAYS: int = 5
PURGE_GAP_BARS: int = 10  # 2× horizon - eliminates overlapping-label bias
N_CV_SPLITS: int = 5
CV_TEST_SIZE_BARS: int = 55
# Minimum feature rows for valid CV: ceil(N_CV_SPLITS*(CV_TEST_SIZE_BARS+PURGE_GAP_BARS) / 0.85)
# = ceil(5*(55+10)/0.85) = ceil(382) = 383 - rounded to 400 for safety margin
MIN_TRAINING_BARS: int = 400

# Model selection skip criteria - COMBINED (mean and std checks)
CV_F1_STD_HARD_LIMIT: float = 0.15
CV_F1_MEAN_MINIMUM: float = 0.52  # below this mean F1 is near-random for binary classification

# LightGBM hyperparameters
SEED: int = 42
LGBM_N_ESTIMATORS: int = 200
LGBM_NUM_LEAVES: int = 31
LGBM_MIN_CHILD_SAMPLES: int = 30  # raised from 20 - small fold sizes need more regularisation
LGBM_LEARNING_RATE: float = 0.05
LGBM_SUBSAMPLE: float = 0.8
LGBM_COLSAMPLE_BYTREE: float = 0.8

# Model persistence
MODEL_RETENTION_DAYS: int = 7
ML_NODE_TIMEOUT_SECONDS: int = 30

# Cost-adjusted label threshold
ROUND_TRIP_COST_BPS: float = 10.0

# Data quality
MIN_PRICE_ROWS: int = 504
MAX_DAILY_RETURN_PCT: float = 50.0

# Outcome evaluation thresholds
MIN_OOS_FOR_DISPLAY: int = 10
MIN_OOS_FOR_DECISIONS: int = 200  # per ticker - not pooled
MIN_OOS_FOR_MEANINGFUL_ACCURACY: int = 100
BASELINE_WINDOW_WEEKS: int = 8
DRIFT_STD_MULTIPLIER: float = 2.0
AGENT_UNDERPERFORM_MARGIN: float = 0.05

# Resolution
MAX_RESOLUTION_ATTEMPTS: int = 5
RESOLUTION_RETRY_DAYS: int = 30

# Recommendation → direction mapping
REC_TO_DIRECTION: dict[str, str | None] = {
    "BUY": "UP",
    "SELL": "DOWN",
    "AVOID": "DOWN",
    "HOLD": None,
}

# Ensemble only: defined here for consumers that reference the constants
ML_CONFIRM_THRESHOLD: float = 0.60
ML_OPPOSE_THRESHOLD: float = 0.40

# Feature columns included in the ML feature matrix.
# ISA_9 (Senkou Span A) excluded: forward-projected 26 bars = look-ahead bias.
# ICS_26 (Chikou) excluded: conservative per design review.
FEATURE_COLS: list[str] = [
    "RSI_14",
    "MACD_12_26_9",
    "MACDs_12_26_9",
    "MACDh_12_26_9",
    "BBU_20_2.0_2.0",
    "BBM_20_2.0_2.0",
    "BBL_20_2.0_2.0",
    "SMA_50",
    "SMA_200",
    "STOCHk_14_3_3",
    "STOCHd_14_3_3",
    "ATRr_14",
    "OBV",
    "WILLR_14",
    "CCI_20_0.015",
    "ADX_14",
    "DMP_14",
    "DMN_14",
    "ITS_9",
    "IKS_26",  # Tenkan + Kijun - NOT forward-shifted
    "STOCHRSIk_14_14_3_3",
    "STOCHRSId_14_14_3_3",
    "AO_5_34",
    "MOM_10",
    "ROC_10",
    "VWMA_20",
    "EMA_10",
    "EMA_20",
    "EMA_30",
    "EMA_50",
    "EMA_100",
    "EMA_200",
    "rolling_52w_high",
    "rolling_52w_low",  # rolling windows, not .tail() snapshots
    "vol_ratio",
    "pct_from_high",
    "pct_from_low",
]
