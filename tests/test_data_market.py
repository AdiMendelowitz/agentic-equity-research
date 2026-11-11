"""Tests for src/data/market.py: get_batch_snapshot, constants."""

from unittest.mock import patch

import pandas as pd
import pytest

from aer.data.cache import _MISSING, TTL_PRICES, Cache
from aer.data.market import (
    CRYPTO_TICKERS,
    INDICES,
    SECTOR_ETFS,
    TTL_MARKET,
    get_batch_snapshot,
)
from aer.utils.cache import _DomainCache


@pytest.fixture
def fresh_cache(tmp_path) -> Cache:
    return Cache(db_path=str(tmp_path / "market.db"))


@pytest.fixture
def fresh_domain_cache(fresh_cache: Cache) -> _DomainCache:
    return _DomainCache(fresh_cache, TTL_PRICES, "data")


def _make_close(tickers: list[str], prices: list[float]) -> pd.DataFrame:
    """Multi-ticker Close DataFrame with MultiIndex columns."""
    idx = pd.date_range("2026-04-25", periods=5, freq="B")
    data = {("Close", t): [p - 1, p - 0.5, p, p + 0.5, p + 1] for t, p in zip(tickers, prices)}
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestConstants:
    def test_ttl_market_15min(self) -> None:
        assert TTL_MARKET == 900

    def test_indices_has_four_entries(self) -> None:
        assert len(INDICES) == 4

    def test_sector_etfs_has_eleven_entries(self) -> None:
        assert len(SECTOR_ETFS) == 11

    def test_crypto_tickers_nonempty(self) -> None:
        assert len(CRYPTO_TICKERS) >= 4


class TestGetBatchSnapshot:
    def test_empty_tickers_returns_empty_dict(self, fresh_domain_cache: _DomainCache) -> None:
        with patch("aer.data.market.data_cache", return_value=fresh_domain_cache):
            assert get_batch_snapshot([]) == {}

    def test_cache_hit_skips_yfinance(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        payload = {"AAPL": {"price": 200.0, "change_pct": 1.0, "change_abs": 2.0}}
        fresh_domain_cache.set("market:batch:AAPL", payload, ttl=TTL_MARKET)
        with patch("aer.data.market.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download") as mock_dl:
                result = get_batch_snapshot(["AAPL"])
        mock_dl.assert_not_called()
        assert result == payload

    def test_single_ticker_flat_columns(self, fresh_domain_cache: _DomainCache) -> None:
        idx = pd.date_range("2026-04-25", periods=5, freq="B")
        df = pd.DataFrame({"Close": [100, 101, 102, 103, 104], "Volume": [1000] * 5}, index=idx)
        with patch("aer.data.market.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", return_value=df):
                result = get_batch_snapshot(["AAPL"])
        assert "AAPL" in result
        assert result["AAPL"]["price"] == pytest.approx(104.0)
        assert result["AAPL"]["change_pct"] == pytest.approx((104 / 103 - 1) * 100)

    def test_multi_ticker_multiindex(self, fresh_domain_cache: _DomainCache) -> None:
        raw = _make_close(["AAPL", "MSFT"], [150.0, 300.0])
        with patch("aer.data.market.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", return_value=raw):
                result = get_batch_snapshot(["AAPL", "MSFT"])
        assert "AAPL" in result
        assert "MSFT" in result
        assert result["AAPL"]["price"] == pytest.approx(151.0)

    def test_multiindex_ticker_case_normalised(self, fresh_domain_cache: _DomainCache) -> None:
        raw = _make_close(["aapl"], [150.0])
        with patch("aer.data.market.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", return_value=raw):
                result = get_batch_snapshot(["AAPL"])
        assert "AAPL" in result

    def test_empty_download_returns_empty_not_cached(
        self, fresh_cache: Cache, fresh_domain_cache: _DomainCache
    ) -> None:
        with patch("aer.data.market.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", return_value=pd.DataFrame()):
                result = get_batch_snapshot(["AAPL"])
        assert result == {}
        assert fresh_cache.get("market:batch:AAPL") is _MISSING

    def test_all_tickers_fail_returns_empty_not_cached(
        self, fresh_cache: Cache, fresh_domain_cache: _DomainCache
    ) -> None:
        idx = pd.date_range("2026-04-25", periods=5, freq="B")
        raw = pd.DataFrame(
            {("Close", "AAPL"): [float("nan")] * 5},
            index=idx,
        )
        raw.columns = pd.MultiIndex.from_tuples(raw.columns)
        with patch("aer.data.market.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", return_value=raw):
                result = get_batch_snapshot(["AAPL"])
        assert result == {}
        assert fresh_cache.get("market:batch:AAPL") is _MISSING

    def test_download_exception_returns_empty(self, fresh_domain_cache: _DomainCache) -> None:
        with patch("aer.data.market.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", side_effect=RuntimeError("network")):
                result = get_batch_snapshot(["AAPL"])
        assert result == {}

    def test_result_cached_on_success(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        idx = pd.date_range("2026-04-25", periods=5, freq="B")
        df = pd.DataFrame({"Close": [100.0, 101, 102, 103, 104], "Volume": [1] * 5}, index=idx)
        with patch("aer.data.market.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", return_value=df):
                get_batch_snapshot(["AAPL"])
        assert fresh_cache.get("market:batch:AAPL") is not None

    def test_change_pct_correct(self, fresh_domain_cache: _DomainCache) -> None:
        idx = pd.date_range("2026-04-25", periods=5, freq="B")
        df = pd.DataFrame({"Close": [100.0, 100, 100, 100, 110], "Volume": [1] * 5}, index=idx)
        with patch("aer.data.market.data_cache", return_value=fresh_domain_cache):
            with patch("yfinance.download", return_value=df):
                result = get_batch_snapshot(["AAPL"])
        assert result["AAPL"]["change_pct"] == pytest.approx(10.0)
        assert result["AAPL"]["change_abs"] == pytest.approx(10.0)
