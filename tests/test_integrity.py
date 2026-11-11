"""Tests for price history integrity validation."""

import numpy as np
import pandas as pd
import pytest

from aer.data.integrity import PriceIntegrityError, validate_price_history


def _make_df(n: int = 600) -> pd.DataFrame:
    """Return a clean DataFrame with n trading-day rows."""
    idx = pd.bdate_range("2023-01-01", periods=n)
    close = pd.Series(np.linspace(100.0, 150.0, n), index=idx)
    volume = pd.Series(np.ones(n) * 1_000_000, index=idx)
    splits = pd.Series(np.zeros(n), index=idx)
    return pd.DataFrame({"Close": close, "Volume": volume, "Stock Splits": splits})


def test_valid_df_passes():
    df = _make_df(600)
    assert validate_price_history(df, "AAPL") is None


def test_too_few_rows_raises():
    df = _make_df(503)
    with pytest.raises(PriceIntegrityError, match="rows"):
        validate_price_history(df, "AAPL")


def test_consecutive_zero_volume_raises():
    df = _make_df(600)
    df.loc[df.index[100:104], "Volume"] = 0  # 4 consecutive zeros
    with pytest.raises(PriceIntegrityError, match="zero-volume"):
        validate_price_history(df, "AAPL")


def test_three_consecutive_zero_volume_passes():
    df = _make_df(600)
    df.loc[df.index[100:103], "Volume"] = 0  # exactly 3 — allowed
    assert validate_price_history(df, "AAPL") is None


def test_price_jump_without_split_raises():
    df = _make_df(600)
    df.loc[df.index[200], "Close"] = df.loc[df.index[199], "Close"] * 2.0  # 100% jump, no split
    with pytest.raises(PriceIntegrityError, match="price jump"):
        validate_price_history(df, "AAPL")


def test_price_jump_with_split_passes():
    df = _make_df(600)
    df.loc[df.index[200], "Close"] = df.loc[df.index[199], "Close"] * 2.0
    df.loc[df.index[200], "Stock Splits"] = 2.0  # split present in window
    assert validate_price_history(df, "AAPL") is None


def test_duplicate_dates_raises():
    df = _make_df(600)
    df = pd.concat([df, df.iloc[[0]]])  # duplicate first row
    with pytest.raises(PriceIntegrityError, match="duplicate"):
        validate_price_history(df, "AAPL")


def test_nan_in_last_ten_close_raises():
    df = _make_df(600)
    df.loc[df.index[-5], "Close"] = float("nan")
    with pytest.raises(PriceIntegrityError, match="NaN"):
        validate_price_history(df, "AAPL")
