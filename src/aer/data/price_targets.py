"""Price target computation — ATR-14, rolling support/resistance. No LLM, no cache."""

import logging
from typing import Any

import pandas as pd
import pandas_ta as ta  # type: ignore

from aer.data.prices import get_raw_history_last_year

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — every threshold named with inline comment)
# ---------------------------------------------------------------------------
NEAR_SUPPORT_PCT = 0.03  # within 3% of 20-day low = near support
ATR_MULTIPLIER = 1.5  # multiplier for ATR buffer entry and stop
ATR_PERIOD = 14  # ATR lookback period
SUPPORT_LOOKBACK = 20  # days for rolling support/resistance
FALLBACK_DISCOUNT = 0.98  # fallback entry = current * 0.98
TARGET_MULTIPLIER = 1.04  # fallback exit = entry * 1.04


def _get_atr14(df: pd.DataFrame) -> float | None:
    """Extract scalar ATR-14 via standalone call. Returns None if insufficient data."""
    if len(df) < ATR_PERIOD:
        return None
    series = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_PERIOD)
    if series is None or series.dropna().empty:
        return None
    return float(series.dropna().iloc[-1])


def compute_price_targets(ticker: str) -> dict[str, Any] | None:
    """Compute entry/exit/stop prices using ATR-14 and rolling support/resistance.

    Returns None if price history is unavailable or empty.
    """
    # ATR, support, resistance, entry and stop are signal computations -> raw path
    # (prices.py:115-124); the adjusted path rescales history on every dividend.
    df = get_raw_history_last_year(ticker)
    if df.empty:
        return None

    current_price = float(df["Close"].iloc[-1])
    atr_val = _get_atr14(df)

    # --- Entry price three-tier logic) ---
    support_20d = float(df["Low"].tail(SUPPORT_LOOKBACK).min())
    if atr_val is not None:
        # Tier 1: near rolling support
        if current_price <= support_20d * (1 + NEAR_SUPPORT_PCT):
            entry_price = max(round(support_20d, 2), 0.01)
            entry_method = "support_level"
        else:
            # Tier 2: ATR buffer below current price
            entry_price = max(round(current_price - ATR_MULTIPLIER * atr_val, 2), 0.01)
            entry_method = "atr_buffer"
    else:
        # Tier 3: fallback percentage discount
        entry_price = max(round(current_price * FALLBACK_DISCOUNT, 2), 0.01)
        entry_method = "fallback_pct"

    # --- Stop loss ---
    if atr_val is not None:
        stop_loss_price = max(round(entry_price - ATR_MULTIPLIER * atr_val, 2), 0.01)
    else:
        stop_loss_price = max(round(entry_price * FALLBACK_DISCOUNT, 2), 0.01)
    if stop_loss_price >= entry_price:
        logger.warning(
            "stop_loss_price %.4f >= entry_price %.4f for %s — possible high-ATR ticker",
            stop_loss_price,
            entry_price,
            ticker,
        )

    # --- Target price ---
    if atr_val is not None:
        risk = entry_price - stop_loss_price
        rr_target = entry_price + 2 * risk  # 2:1 risk/reward
        resistance_20d = float(df["High"].tail(SUPPORT_LOOKBACK).max())
        if resistance_20d < rr_target:  # resistance cap
            target_price = round(resistance_20d, 2)
            target_method = "resistance"
        else:
            target_price = round(rr_target, 2)
            target_method = "risk_reward_2_1"
    else:
        target_price = round(entry_price * TARGET_MULTIPLIER, 2)  # fallback
        target_method = "fallback_pct"

    # --- Risk/reward ratio ---
    denominator = entry_price - stop_loss_price
    risk_reward_ratio = round((target_price - entry_price) / denominator, 4) if denominator > 0 else 0.0

    return {
        "entry_price": entry_price,
        "entry_method": entry_method,
        "target_price": target_price,
        "target_method": target_method,
        "stop_loss_price": stop_loss_price,
        "risk_reward_ratio": risk_reward_ratio,
    }
