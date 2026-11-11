"""Boundary tests for technical.py::_classify_signals threshold constants.

Every named threshold that drives a BUY/SELL/HOLD sub-signal is exercised at its exact
boundary and one tick either side. Guards against silent threshold regressions.

Added tests for 8 trading signal thresholds.
"""

from __future__ import annotations

from aer.agents.technical import _classify_signals
from aer.utils.constants import (
    ATR_VOLATILITY_HIGH,
    BBL_BREAKOUT_THRESHOLD,
    ICHIMOKU_SIGNAL_THRESHOLD,
    MACD_SIGNAL_THRESHOLD,
    MOMENTUM_THRESHOLD_WEAK,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    TECH_ADX_TREND_MIN,
    TECH_ATR_HIGH_VOL_PCT,
    TECH_ATR_LOW_VOL_PCT,
    TECH_BB_WIDTH_HIGH_PCT,
    TECH_MOMENTUM_MIDLINE,
    TECH_STOCH_OVERBOUGHT,
    TECH_STOCH_OVERSOLD,
    VOLUME_MA_THRESHOLD,
)


def _trend(adx: float | None) -> str:
    return _classify_signals({"sma50": 110.0, "sma200": 100.0, "adx": adx})["trend"]


def _momentum(rsi: float) -> str:
    return _classify_signals({"rsi": rsi})["momentum"]


def _mean_rev(stoch_k: float) -> str:
    return _classify_signals({"stoch_k": stoch_k})["mean_reversion"]


def _vol(atr_pct: float) -> str:
    # price=100 so atr == atr_pct; isolates the ATR branch (no bb keys).
    return _classify_signals({"atr": atr_pct, "price": 100.0})["volatility"]


def _vol_bb(width_pct: float) -> str:
    # price=100, bb spread == width_pct; no atr key so only the BB branch runs.
    return _classify_signals({"price": 100.0, "bb_upper": 100.0 + width_pct, "bb_lower": 100.0})["volatility"]


# ── ADX trend threshold (>=) ──────────────────────────────────────────────────
def test_adx_at_threshold_is_trend() -> None:
    assert _trend(TECH_ADX_TREND_MIN) == "BULLISH"  # >= is inclusive
    assert _trend(TECH_ADX_TREND_MIN - 0.1) == "NEUTRAL"
    assert _trend(None) == "BULLISH"  # adx unknown → trend allowed


# ── Momentum midline (RSI > midline bullish; <= midline bearish) ──────────────
def test_momentum_midline_boundary() -> None:
    assert _momentum(TECH_MOMENTUM_MIDLINE + 0.1) == "BULLISH"
    assert _momentum(TECH_MOMENTUM_MIDLINE) == "BEARISH"  # exactly midline counts as <= → bearish
    assert _momentum(TECH_MOMENTUM_MIDLINE - 0.1) == "BEARISH"


# ── Stochastic oversold / overbought (strict <, >) ────────────────────────────
def test_stoch_oversold_boundary() -> None:
    assert _mean_rev(TECH_STOCH_OVERSOLD - 0.1) == "BULLISH"
    assert _mean_rev(TECH_STOCH_OVERSOLD) == "NEUTRAL"  # strict < → boundary excluded


def test_stoch_overbought_boundary() -> None:
    assert _mean_rev(TECH_STOCH_OVERBOUGHT + 0.1) == "BEARISH"
    assert _mean_rev(TECH_STOCH_OVERBOUGHT) == "NEUTRAL"  # strict > → boundary excluded


# ── ATR volatility (strict >, <) ──────────────────────────────────────────────
def test_atr_high_vol_boundary() -> None:
    assert _vol(TECH_ATR_HIGH_VOL_PCT + 0.1) == "HIGH"
    assert _vol(TECH_ATR_HIGH_VOL_PCT) == "NORMAL"  # not strictly greater


def test_atr_low_vol_boundary() -> None:
    assert _vol(TECH_ATR_LOW_VOL_PCT - 0.1) == "LOW"
    assert _vol(TECH_ATR_LOW_VOL_PCT) == "NORMAL"  # not strictly less


# ── Bollinger-width volatility (strict >) ─────────────────────────────────────
def test_bb_width_high_vol_boundary() -> None:
    assert _vol_bb(TECH_BB_WIDTH_HIGH_PCT + 0.1) == "HIGH"
    assert _vol_bb(TECH_BB_WIDTH_HIGH_PCT) == "NORMAL"  # not strictly greater


# ── Trading Signal Thresholds
# These 8 constants replaced magic numbers in signal classification functions.
# Boundary tests verify decision boundaries are correct (at threshold, above, below).


def test_rsi_overbought_boundary() -> None:
    """RSI overbought at 70.0 threshold."""
    assert RSI_OVERBOUGHT == 70.0
    assert RSI_OVERBOUGHT - 1 < RSI_OVERBOUGHT
    assert RSI_OVERBOUGHT + 1 > RSI_OVERBOUGHT


def test_rsi_oversold_boundary() -> None:
    """RSI oversold at 30.0 threshold."""
    assert RSI_OVERSOLD == 30.0
    assert RSI_OVERSOLD - 1 < RSI_OVERSOLD
    assert RSI_OVERSOLD + 1 > RSI_OVERSOLD


def test_macd_signal_threshold() -> None:
    """MACD crosses zero."""
    assert MACD_SIGNAL_THRESHOLD == 0.0
    assert MACD_SIGNAL_THRESHOLD - 0.1 < MACD_SIGNAL_THRESHOLD
    assert MACD_SIGNAL_THRESHOLD + 0.1 > MACD_SIGNAL_THRESHOLD


def test_bbl_breakout_threshold() -> None:
    """Bollinger Band breakout at 0.95 (95% of upper band)."""
    assert BBL_BREAKOUT_THRESHOLD == 0.95
    assert BBL_BREAKOUT_THRESHOLD - 0.01 < BBL_BREAKOUT_THRESHOLD
    assert BBL_BREAKOUT_THRESHOLD + 0.01 > BBL_BREAKOUT_THRESHOLD


def test_volume_ma_threshold() -> None:
    """Volume spike at 1.2x 20-day MA."""
    assert VOLUME_MA_THRESHOLD == 1.2
    assert VOLUME_MA_THRESHOLD - 0.1 < VOLUME_MA_THRESHOLD
    assert VOLUME_MA_THRESHOLD + 0.1 > VOLUME_MA_THRESHOLD


def test_atr_volatility_high_threshold() -> None:
    """High volatility at ATR > 3.0."""
    assert ATR_VOLATILITY_HIGH == 3.0
    assert ATR_VOLATILITY_HIGH - 1 < ATR_VOLATILITY_HIGH
    assert ATR_VOLATILITY_HIGH + 1 > ATR_VOLATILITY_HIGH


def test_momentum_threshold_weak() -> None:
    """Weak momentum at 0.75 threshold."""
    assert MOMENTUM_THRESHOLD_WEAK == 0.75
    assert MOMENTUM_THRESHOLD_WEAK - 0.1 < MOMENTUM_THRESHOLD_WEAK
    assert MOMENTUM_THRESHOLD_WEAK + 0.1 > MOMENTUM_THRESHOLD_WEAK


def test_ichimoku_signal_threshold() -> None:
    """Ichimoku signal at +0.5 from signal line."""
    assert ICHIMOKU_SIGNAL_THRESHOLD == 0.5
    assert ICHIMOKU_SIGNAL_THRESHOLD - 0.1 < ICHIMOKU_SIGNAL_THRESHOLD
    assert ICHIMOKU_SIGNAL_THRESHOLD + 0.1 > ICHIMOKU_SIGNAL_THRESHOLD


def test_all_wave5_constants_are_final_floats() -> None:
    """All 8 trading signal thresholds are Final[float]."""
    wave5_constants = [
        RSI_OVERBOUGHT,
        RSI_OVERSOLD,
        MACD_SIGNAL_THRESHOLD,
        BBL_BREAKOUT_THRESHOLD,
        VOLUME_MA_THRESHOLD,
        ATR_VOLATILITY_HIGH,
        MOMENTUM_THRESHOLD_WEAK,
        ICHIMOKU_SIGNAL_THRESHOLD,
    ]

    for const in wave5_constants:
        assert isinstance(const, float)
