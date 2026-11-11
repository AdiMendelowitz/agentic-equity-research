"""ML feature matrix construction.

Single-pass approach: compute all pandas-ta indicators with append=True on df.copy(),
then build one row per trading day by reading column values at each bar's index.
This mirrors harness._compute_signals() performance characteristics (~1.5s for 756 bars).

Design review fixes applied:
- B1: Uses get_price_history_raw (auto_adjust=False), not get_price_history
- B2: Per-bar column reads, not .iloc[-1] snapshot broadcast
- H2: Split guard - labels for bars with splits in [t, t+5] window are NaN
- M1: ISA_9 excluded (Senkou Span A - forward-shifted 26 bars in pandas-ta)
- L1: rolling_52w_high/low via .rolling(252) not .tail(252)
"""

import logging
from typing import Any

import pandas as pd
import pandas_ta as ta  # noqa: F401 - registers df.ta accessor

from aer.data.integrity import validate_price_history
from aer.data.prices import get_price_history_raw
from aer.ml.constants import (
    FEATURE_COLS,
    MIN_TRAINING_BARS,
    PREDICTION_HORIZON_DAYS,
    ROUND_TRIP_COST_BPS,
)
from aer.utils.trading_calendar import add_trading_days

logger = logging.getLogger(__name__)


def build_feature_matrix(ticker: str, period: str = "3y", prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a labeled feature DataFrame for ML training and prediction.

    Returns a DataFrame indexed by trading date with:
    - Feature columns (see FEATURE_COLS in constants.py)
    - 'label' column: 1 if 5-day forward return > ROUND_TRIP_COST_BPS/10000, 0 otherwise
      (NaN for the last PREDICTION_HORIZON_DAYS rows - no forward return available yet)

    Callers:
    - train_and_predict() uses dropna(subset=["label"]) for training data
    - The last row (NaN label) provides live features for prediction
    """
    df = get_price_history_raw(ticker, period=period) if prices is None else prices
    validate_price_history(df, ticker)

    df = df.copy()

    # Single-pass indicator computation (append=True mutates df - that's the intent)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.willr(length=14, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)
    df.ta.ao(append=True)
    df.ta.mom(length=10, append=True)
    df.ta.roc(append=True)
    df.ta.vwma(length=20, append=True)

    # Ichimoku: ISA_9 (Senkou Span A) is forward-projected 26 bars - excluded.
    # include_chikou=False suppresses ICS_26. Only merge ITS_9 and IKS_26.
    try:
        ichi, _ = df.ta.ichimoku(tenkan=9, kijun=26, senkou=52, include_chikou=False)
        if ichi is not None:
            for col in ["ITS_9", "IKS_26"]:
                if col in ichi.columns:
                    df[col] = ichi[col]
    except Exception as e:
        logger.warning("Ichimoku computation failed for %s: %s", ticker, e)

    # EMA suite - guard len(df) >= ema_len (pandas-ta returns full df silently if shorter)
    for ema_len in [10, 20, 30, 50, 100, 200]:
        if len(df) >= ema_len:
            df.ta.ema(length=ema_len, append=True)

    # Rolling-based derived features - correct for per-bar feature matrix
    # .tail(252) would give the same snapshot for every row; .rolling(252) gives per-bar values
    df["rolling_52w_high"] = df["Close"].rolling(252, min_periods=20).max()
    df["rolling_52w_low"] = df["Close"].rolling(252, min_periods=20).min()
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20, min_periods=5).mean()
    df["pct_from_high"] = (df["Close"] / df["rolling_52w_high"] - 1) * 100
    df["pct_from_low"] = (df["Close"] / df["rolling_52w_low"] - 1) * 100

    # Select only feature columns that were successfully computed
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    # Build label: cost-adjusted binary direction label for each bar
    # this label is NOT src/data/opinions.py's forward_return_5d. Here: a binary, thresholded
    # at ROUND_TRIP_COST_BPS, on the dividend-UNADJUSTED path, with an explicit split guard. There:
    # an uncosted continuous total return on the dividend-ADJUSTED path(B)). Same nominal
    # horizon, two definitions, both intentional; do not reconcile one into the other.
    # Guard: bars where a stock split falls within [t+1, t+PREDICTION_HORIZON_DAYS] get NaN label
    splits_col = df.get("Stock Splits", pd.Series(0.0, index=df.index))
    splits_arr = splits_col.values
    close_arr = df["Close"].values
    dates = df.index.tolist()
    date_to_idx = {d.date(): i for i, d in enumerate(dates)}

    labels: dict[Any, float] = {}
    threshold = ROUND_TRIP_COST_BPS / 10_000
    for i, dt in enumerate(dates):
        target_date = add_trading_days(dt.date(), PREDICTION_HORIZON_DAYS)
        j = date_to_idx.get(target_date)
        if j is None:
            continue
        # Skip bars with any split between i (exclusive) and j (inclusive)
        if splits_arr[i + 1 : j + 1].sum() > 0:
            continue
        fwd_return = close_arr[j] / close_arr[i] - 1
        labels[dt] = float(int(fwd_return > threshold))

    df["label"] = pd.Series(labels)

    # Retain only feature columns + label; drop rows where ANY feature is NaN
    result = df[available_features + ["label"]].dropna(subset=available_features)

    assert df.index.is_monotonic_increasing, f"{ticker}: price DataFrame index is not monotonic"

    # Sanity: no OHLCV columns in output
    forbidden = {"Open", "High", "Low", "Close", "Volume", "Stock Splits"}
    assert not forbidden.intersection(result.columns)

    train_rows = len(result.dropna(subset=["label"]))
    logger.info(
        "%s: feature matrix built — %d labeled training rows, %d features",
        ticker,
        train_rows,
        len(available_features),
    )

    if train_rows < MIN_TRAINING_BARS:
        raise ValueError(f"{ticker}: only {train_rows} labeled training rows — need at least {MIN_TRAINING_BARS}")

    return result
