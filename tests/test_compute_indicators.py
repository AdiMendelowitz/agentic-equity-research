"""Tests for aer.indicators.technical.compute_indicators()."""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int) -> pd.DataFrame:
    """Build synthetic OHLCV DataFrame with n bars."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
    high = close + abs(np.random.randn(n) * 0.3)
    low = close - abs(np.random.randn(n) * 0.3)
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.randint(1_000_000, 5_000_000, n), index=dates)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


REQUIRED_KEYS = [
    "price",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "sma50",
    "sma200",
    "stoch_k",
    "stoch_d",
    "atr",
    "obv",
    "willr",
    "cci",
    "adx",
    "adx_pos",
    "adx_neg",
    "ichi_tenkan",
    "ichi_kijun",
    "ichi_senkou_a",
    "ichi_chikou",
    "stochrsi_k",
    "stochrsi_d",
    "ao",
    "mom",
    "roc",
    "vwma",
    "ema_10",
    "ema_20",
    "ema_30",
    "ema_50",
    "ema_100",
    "ema_200",
    "vol_ratio",
    "high_52w",
    "low_52w",
    "pct_from_high",
    "pct_from_low",
]


def test_all_required_keys_present_on_252_bars() -> None:
    from aer.indicators.technical import compute_indicators

    df = _make_ohlcv(252)
    result = compute_indicators(df.copy())
    missing = [k for k in REQUIRED_KEYS if k not in result]
    assert not missing, f"Missing keys: {missing}"


def test_no_ichi_cloud_keys_in_result() -> None:
    from aer.indicators.technical import compute_indicators

    df = _make_ohlcv(252)
    result = compute_indicators(df.copy())
    cloud_keys = [k for k in result if "cloud" in k.lower()]
    assert not cloud_keys, f"ichi_cloud keys leaked into result: {cloud_keys}"


def test_key_indicators_are_floats_on_252_bars() -> None:
    from aer.indicators.technical import compute_indicators

    df = _make_ohlcv(252)
    result = compute_indicators(df.copy())
    for key in ["price", "rsi", "macd", "bb_upper", "sma50", "adx", "atr", "obv", "ema_50"]:
        assert result[key] is not None, f"{key} is None on 252 bars"
        assert isinstance(result[key], float), f"{key} is not float: {type(result[key])}"


def test_ema_200_none_on_short_dataframe() -> None:
    from aer.indicators.technical import compute_indicators

    df = _make_ohlcv(150)
    result = compute_indicators(df.copy())
    assert result["ema_200"] is None, "ema_200 must be None when len(df) < 200"


def test_high_52w_and_low_52w_none_on_short_dataframe() -> None:
    from aer.indicators.technical import compute_indicators

    df = _make_ohlcv(17)
    result = compute_indicators(df.copy())
    assert result["high_52w"] is None, "high_52w must be None when len(df) < 252"
    assert result["low_52w"] is None, "low_52w must be None when len(df) < 252"


def test_price_key_is_last_close() -> None:
    from aer.indicators.technical import compute_indicators

    df = _make_ohlcv(252)
    result = compute_indicators(df.copy())
    assert result["price"] == pytest.approx(float(df["Close"].iloc[-1]))


def test_importable_from_package() -> None:
    import aer.indicators  # noqa: F401
    from aer.indicators.technical import compute_indicators  # noqa: F401

    assert callable(compute_indicators)
