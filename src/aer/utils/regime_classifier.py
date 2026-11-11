"""Regime classifier — pure Python, no LLM, no I/O.

Returns a full context dict so computation parameters are embedded in the
recommendation record, preventing mixed-window populations in conditioned metrics.
"""

from aer.utils.constants import (
    REGIME_ADX_TRENDING,
    REGIME_ADX_VOLATILE,
    REGIME_ADX_WINDOW,
    REGIME_ATR_VOLATILE_PCT,
    REGIME_ATR_WINDOW,
)


def classify_regime(
    atr_pct: float,
    adx: float,
    atr_window: int = REGIME_ATR_WINDOW,
    adx_window: int = REGIME_ADX_WINDOW,
) -> dict:
    """Classify market regime from ATR% and ADX.

    Returns:
        dict with keys: regime_label, atr_pct, adx, atr_window, adx_window.
        regime_label is "volatile" | "trending" | "ranging".
    """
    if adx > REGIME_ADX_VOLATILE and atr_pct > REGIME_ATR_VOLATILE_PCT:
        label = "volatile"
    elif adx > REGIME_ADX_TRENDING:
        label = "trending"
    else:
        label = "ranging"
    return {
        "regime_label": label,
        "atr_pct": atr_pct,
        "adx": adx,
        "atr_window": atr_window,
        "adx_window": adx_window,
    }
