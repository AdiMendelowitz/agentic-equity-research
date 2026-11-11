"""Tests for aer.data.price_targets."""

from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_ohlcv(n: int) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
    high = close + abs(np.random.randn(n) * 0.3)
    low = close - abs(np.random.randn(n) * 0.3)
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.randint(1_000_000, 5_000_000, n), index=dates)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


def test_compute_price_targets_returns_all_keys() -> None:
    from aer.data.price_targets import compute_price_targets

    with patch("aer.data.price_targets.get_raw_history_last_year", return_value=_make_ohlcv(252)):
        result = compute_price_targets("AAPL")
    assert result is not None
    for key in ("entry_price", "entry_method", "target_price", "target_method", "stop_loss_price", "risk_reward_ratio"):
        assert key in result, f"Missing key: {key}"


def test_returns_none_on_empty_dataframe() -> None:
    from aer.data.price_targets import compute_price_targets

    with patch("aer.data.price_targets.get_raw_history_last_year", return_value=pd.DataFrame()):
        result = compute_price_targets("AAPL")
    assert result is None


def test_atr_fallback_when_fewer_than_14_bars() -> None:
    from aer.data.price_targets import compute_price_targets

    with patch("aer.data.price_targets.get_raw_history_last_year", return_value=_make_ohlcv(10)):
        result = compute_price_targets("AAPL")
    assert result is not None
    assert result["entry_method"] == "fallback_pct"
    assert result["target_method"] == "fallback_pct"


def test_risk_reward_ratio_is_positive() -> None:
    from aer.data.price_targets import compute_price_targets

    with patch("aer.data.price_targets.get_raw_history_last_year", return_value=_make_ohlcv(252)):
        result = compute_price_targets("AAPL")
    assert result is not None
    assert result["risk_reward_ratio"] >= 1.0


def test_entry_price_near_support() -> None:
    """When current price is within NEAR_SUPPORT_PCT of 20-day low, entry_method == 'support_level'."""
    from aer.data.price_targets import NEAR_SUPPORT_PCT, SUPPORT_LOOKBACK, compute_price_targets

    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    # Flat price at 100, low at 98 — current (100) is within 3% of 20-day low (98)
    close = pd.Series([100.0] * n, index=dates)
    high = pd.Series([102.0] * n, index=dates)
    low = pd.Series([98.0] * n, index=dates)
    open_ = close.copy()
    volume = pd.Series([1_000_000] * n, index=dates)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})

    # Verify the fixture actually triggers the near-support branch
    support_20d = low.tail(SUPPORT_LOOKBACK).min()  # 98.0
    assert close.iloc[-1] <= support_20d * (1 + NEAR_SUPPORT_PCT), "Fixture did not trigger near-support branch"

    with patch("aer.data.price_targets.get_raw_history_last_year", return_value=df):
        result = compute_price_targets("AAPL")

    assert result is not None
    assert result["entry_method"] == "support_level"
    assert result["entry_price"] == round(float(support_20d), 2)


def test_target_price_capped_by_resistance() -> None:
    """When 20-day resistance < uncapped 2:1 rr_target, target_method == 'resistance'."""
    from aer.data.price_targets import SUPPORT_LOOKBACK, compute_price_targets

    n = 50
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    # Wide ATR (big H-L spread) so that 2:1 target > resistance cap.
    # Current close=100, tight resistance at 101, large ATR (~21) from high-low spread of 21.
    close = pd.Series([100.0] * n, index=dates)
    high = pd.Series([101.0] * n, index=dates)  # 20-day max high = 101 (tight resistance)
    low = pd.Series([80.0] * n, index=dates)  # ATR will be ~20+ (high-low spread = 21)
    open_ = close.copy()
    volume = pd.Series([1_000_000] * n, index=dates)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})

    with patch("aer.data.price_targets.get_raw_history_last_year", return_value=df):
        result = compute_price_targets("AAPL")

    assert result is not None
    # With ATR ~21 and atr_buffer entry well below current, the uncapped 2:1 target exceeds 101.
    assert result["target_method"] == "resistance", (
        f"Expected 'resistance' but got '{result['target_method']}'. "
        f"target_price={result['target_price']}, resistance_20d={high.tail(SUPPORT_LOOKBACK).max()}"
    )
    assert result["target_price"] <= high.tail(SUPPORT_LOOKBACK).max() + 0.01
