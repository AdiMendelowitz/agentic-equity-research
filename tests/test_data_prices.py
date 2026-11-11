"""Tests for src/data/prices.py: caching, NaN handling, ticker normalisation."""

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from aer.data.cache import TTL_PRICES, Cache
from aer.data.prices import (
    _cacheable_to_df,
    _clean_value,
    _df_to_cacheable,
    get_current_price,
    get_price_history,
)
from aer.utils.cache import _DomainCache


@pytest.fixture
def fresh_cache(tmp_path) -> Cache:
    return Cache(db_path=str(tmp_path / "prices.db"))


@pytest.fixture
def fresh_domain_cache(fresh_cache: Cache) -> _DomainCache:
    return _DomainCache(fresh_cache, TTL_PRICES, "data")


def _ohlcv(n: int = 10, base: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": [base] * n,
            "High": [base + 1] * n,
            "Low": [base - 1] * n,
            "Close": [base + i * 0.5 for i in range(n)],
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


class TestCleanValue:
    def test_nan_becomes_none(self) -> None:
        assert _clean_value(float("nan")) is None

    def test_normal_float_unchanged(self) -> None:
        assert _clean_value(3.14) == pytest.approx(3.14)

    def test_int_unchanged(self) -> None:
        assert _clean_value(42) == 42

    def test_string_unchanged(self) -> None:
        assert _clean_value("hello") == "hello"

    def test_numpy_scalar_converted(self) -> None:
        import numpy as np

        v = np.float64(1.5)
        result = _clean_value(v)
        assert result == pytest.approx(1.5)
        assert isinstance(result, float)


class TestDfCacheable:
    def test_roundtrip_preserves_values(self) -> None:
        df = _ohlcv(5)
        restored = _cacheable_to_df(_df_to_cacheable(df))
        assert list(restored.columns) == list(df.columns)
        assert len(restored) == len(df)
        assert restored["Close"].iloc[-1] == pytest.approx(df["Close"].iloc[-1])

    def test_nan_serialised_as_none(self) -> None:
        df = _ohlcv(3)
        df.loc[df.index[1], "Close"] = float("nan")
        cacheable = _df_to_cacheable(df)
        assert cacheable["data"][1][3] is None  # Close column index = 3

    def test_no_nan_strings_in_output(self) -> None:
        df = _ohlcv(3)
        df.loc[df.index[0], "Volume"] = float("nan")
        import json

        serialised = json.dumps(_df_to_cacheable(df))
        assert "nan" not in serialised.lower()


class TestGetPriceHistory:
    def test_cache_hit_skips_download(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        df = _ohlcv(5)
        fresh_domain_cache.set("prices:ohlcv:AAPL:1y:1d", _df_to_cacheable(df))
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download") as mock_dl:
                result = get_price_history("AAPL")
        mock_dl.assert_not_called()
        assert len(result) == 5

    def test_ticker_uppercased_for_cache_key(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        df = _ohlcv(5)
        fresh_domain_cache.set("prices:ohlcv:AAPL:1y:1d", _df_to_cacheable(df))
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download") as mock_dl:
                get_price_history("aapl")
        mock_dl.assert_not_called()

    def test_empty_download_returns_empty_df(self, fresh_domain_cache: _DomainCache) -> None:
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", return_value=pd.DataFrame()):
                result = get_price_history("AAPL")
        assert result.empty

    def test_multiindex_columns_flattened(self, fresh_domain_cache: _DomainCache) -> None:
        df = _ohlcv(5)
        multi_df = df.copy()
        multi_df.columns = pd.MultiIndex.from_tuples([(c, "AAPL") for c in df.columns])
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", return_value=multi_df):
                result = get_price_history("AAPL")
        assert "Close" in result.columns
        assert not isinstance(result.columns, pd.MultiIndex)

    def test_result_cached_after_download(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        df = _ohlcv(5)
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", return_value=df):
                get_price_history("AAPL")
        assert fresh_cache.get("prices:ohlcv:AAPL:1y:1d") is not None

    def test_all_nan_rows_dropped(self, fresh_domain_cache: _DomainCache) -> None:
        df = _ohlcv(5)
        df.loc[df.index[2]] = float("nan")
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", return_value=df):
                result = get_price_history("AAPL")
        assert len(result) == 4


class TestGetCurrentPrice:
    def test_cache_hit(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        fresh_domain_cache.set("prices:current:AAPL", 199.5)
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            assert get_current_price("AAPL") == pytest.approx(199.5)

    def test_ticker_uppercased(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        fresh_domain_cache.set("prices:current:AAPL", 199.5)
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            assert get_current_price("aapl") == pytest.approx(199.5)

    def test_falls_back_to_history_when_fast_info_none(self, fresh_domain_cache: _DomainCache) -> None:
        df = _ohlcv(5, base=150.0)
        mock_tk = MagicMock()
        mock_tk.fast_info.last_price = None
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.Ticker", return_value=mock_tk):
                with patch("aer.data.prices.get_price_history", return_value=df):
                    price = get_current_price("AAPL")
        assert price == pytest.approx(df["Close"].iloc[-1])

    def test_returns_none_when_all_sources_fail(self, fresh_domain_cache: _DomainCache) -> None:
        mock_tk = MagicMock()
        mock_tk.fast_info.last_price = None
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.Ticker", return_value=mock_tk):
                with patch("aer.data.prices.get_price_history", return_value=pd.DataFrame()):
                    price = get_current_price("AAPL")
        assert price is None

    def test_history_fallback_skips_trailing_nan_close(self, fresh_domain_cache: _DomainCache) -> None:
        """A pre-market forming row (trailing NaN Close) must not surface a NaN price."""
        dates = pd.date_range("2026-06-03", periods=3, freq="B")
        df = pd.DataFrame(
            {
                "Open": [150.0, 151.0, 152.0],
                "High": [1, 2, 3],
                "Low": [1, 2, 3],
                "Close": [150.0, 151.0, math.nan],
                "Volume": [1, 1, 0],
            },
            index=dates,
        )
        mock_tk = MagicMock()
        mock_tk.fast_info.last_price = None
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.Ticker", return_value=mock_tk):
                with patch("aer.data.prices.get_price_history", return_value=df):
                    price = get_current_price("AAPL")
        assert price == pytest.approx(151.0)

    def test_nan_from_fast_info_not_cached_or_returned(
        self, fresh_cache: Cache, fresh_domain_cache: _DomainCache
    ) -> None:
        """A NaN from fast_info.last_price must be treated as unavailable, not cached."""
        mock_tk = MagicMock()
        mock_tk.fast_info.last_price = math.nan
        with patch("aer.data.prices.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.Ticker", return_value=mock_tk):
                with patch("aer.data.prices.get_price_history", return_value=pd.DataFrame()):
                    price = get_current_price("AAPL")
        assert price is None
        assert fresh_domain_cache.get("prices:current:AAPL") is None  # NaN never cached
