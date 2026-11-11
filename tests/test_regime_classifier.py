"""Tests for src/utils/regime_classifier.py — 7 cases covering all branches and boundary conditions."""

from aer.utils.constants import REGIME_ADX_WINDOW, REGIME_ATR_WINDOW
from aer.utils.regime_classifier import classify_regime


def test_volatile_high_adx_high_atr() -> None:
    result = classify_regime(atr_pct=2.0, adx=30)
    assert result["regime_label"] == "volatile"


def test_trending_high_adx_low_atr() -> None:
    # ADX > REGIME_ADX_TRENDING (20) but ATR% below volatile threshold — trending
    result = classify_regime(atr_pct=1.0, adx=22)
    assert result["regime_label"] == "trending"


def test_ranging_low_adx() -> None:
    result = classify_regime(atr_pct=0.8, adx=15)
    assert result["regime_label"] == "ranging"


def test_boundary_adx_trending() -> None:
    # ADX == REGIME_ADX_TRENDING (20): condition is >, so exactly 20 is NOT trending
    result = classify_regime(atr_pct=0.8, adx=20)
    assert result["regime_label"] == "ranging"


def test_boundary_adx_volatile() -> None:
    # ADX == REGIME_ADX_VOLATILE (25): condition is >, so exactly 25 is NOT volatile → trending
    result = classify_regime(atr_pct=2.0, adx=25)
    assert result["regime_label"] == "trending"


def test_returns_dict_with_context() -> None:
    result = classify_regime(atr_pct=1.2, adx=18)
    assert set(result.keys()) == {"regime_label", "atr_pct", "adx", "atr_window", "adx_window"}
    assert result["atr_pct"] == 1.2
    assert result["adx"] == 18
    assert result["atr_window"] == REGIME_ATR_WINDOW
    assert result["adx_window"] == REGIME_ADX_WINDOW


def test_custom_windows_embedded_in_result() -> None:
    result = classify_regime(atr_pct=1.0, adx=18, atr_window=5, adx_window=7)
    assert result["atr_window"] == 5
    assert result["adx_window"] == 7
