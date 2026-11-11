"""Tests for FRED macro data wrapper: get_series, get_latest, get_macro_snapshot."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from aer.data.cache import TTL_PRICES, Cache
from aer.utils.cache import _DomainCache


@pytest.fixture
def fresh_cache(tmp_path) -> Cache:
    c = Cache(db_path=str(tmp_path / "macro.db"))
    return c


def _patch_cache(monkeypatch, cache: Cache) -> None:
    domain_cache = _DomainCache(cache, TTL_PRICES, "data")
    monkeypatch.setattr("aer.data.macro.data_cache", lambda: domain_cache)


def _mock_fred(value: float = 5.0, periods: int = 14) -> MagicMock:
    m = MagicMock()
    m.get_series.return_value = pd.Series(
        [value] * periods,
        index=pd.date_range("2023-01-01", periods=periods, freq="ME"),
    )
    return m


class TestSeriesConstants:
    def test_series_has_expected_keys(self) -> None:
        from aer.data.macro import SERIES

        # cpi is fetched via hardcoded CPIAUCSL in get_macro_snapshot (derived, not in SERIES)
        for key in (
            "fed_funds_rate",
            "unemployment",
            "treasury_10y",
            "treasury_2y",
            "yield_curve",
            "gdp",
            "pce",
            "vix",
            "sp500",
            "hy_spread",
            "ig_spread",
            "real_yield_10y",
            "breakeven_10y",
        ):
            assert key in SERIES

    def test_fed_funds_maps_to_fedfunds(self) -> None:
        from aer.data.macro import SERIES

        assert SERIES["fed_funds_rate"] == "FEDFUNDS"
        assert SERIES["treasury_10y"] == "DGS10"
        assert SERIES["vix"] == "VIXCLS"


class TestGetSeries:
    def test_returns_pandas_series(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)
        with patch("aer.data.macro._get_fred", return_value=_mock_fred(5.33, 5)):
            from aer.data.macro import get_series

            result = get_series("FEDFUNDS", periods=5)
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_series_name_set_on_cache_hit(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)
        with patch("aer.data.macro._get_fred", return_value=_mock_fred(5.33, 5)):
            from aer.data.macro import get_series

            get_series("FEDFUNDS", periods=5)  # populates cache
            result = get_series("FEDFUNDS", periods=5)  # cache hit → name explicitly set
        assert result.name == "FEDFUNDS"

    def test_respects_periods_limit(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = pd.Series(
            range(100), index=pd.date_range("2023-01-01", periods=100, freq="ME")
        )
        with patch("aer.data.macro._get_fred", return_value=mock_fred):
            from aer.data.macro import get_series

            result = get_series("FEDFUNDS", periods=10)
        assert len(result) == 10

    def test_nan_values_dropped(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = pd.Series(
            [5.0, np.nan, 5.5, np.nan],
            index=pd.date_range("2024-01-01", periods=4, freq="ME"),
        )
        with patch("aer.data.macro._get_fred", return_value=mock_fred):
            from aer.data.macro import get_series

            result = get_series("FEDFUNDS", periods=4)
        assert not result.isna().any()
        assert len(result) == 2

    def test_second_call_uses_cache(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)
        mock_fred = _mock_fred(5.33, 5)
        with patch("aer.data.macro._get_fred", return_value=mock_fred):
            from aer.data.macro import get_series

            get_series("FEDFUNDS", periods=5)
            get_series("FEDFUNDS", periods=5)
        assert mock_fred.get_series.call_count == 1


class TestGetLatest:
    def test_returns_float(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)
        with patch("aer.data.macro._get_fred", return_value=_mock_fred(5.33)):
            from aer.data.macro import get_latest

            result = get_latest("FEDFUNDS")
        assert isinstance(result, float)
        assert result == pytest.approx(5.33)

    def test_returns_none_on_exception(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)
        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = Exception("FRED unavailable")
        with patch("aer.data.macro._get_fred", return_value=mock_fred):
            from aer.data.macro import get_latest

            result = get_latest("FEDFUNDS")
        assert result is None

    def test_returns_none_for_empty_series(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = pd.Series([], dtype=float)
        with patch("aer.data.macro._get_fred", return_value=mock_fred):
            from aer.data.macro import get_latest

            result = get_latest("FEDFUNDS")
        assert result is None


class TestGetMacroSnapshot:
    def test_snapshot_has_all_required_keys(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)
        with patch("aer.data.macro._get_fred", return_value=_mock_fred(5.0, 14)):
            from aer.data.macro import get_macro_snapshot

            snap = get_macro_snapshot()
        for key in (
            "fed_funds_rate",
            "unemployment",
            "treasury_10y",
            "treasury_2y",
            "yield_curve",
            "gdp",
            "pce",
            "vix",
            "sp500",
            "cpi",
            "cpi_yoy_pct",
        ):
            assert key in snap, f"Missing key: {key}"

    def test_cpi_yoy_calculation(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)
        # With 14 values, iloc[-13] = index 1, iloc[-1] = index 13.
        # YoY = (306 / 300 - 1) * 100 = 2.0%
        vals = [999.0, 300.0] + [302.0] * 11 + [306.0]  # index 1 = 300.0, index 13 = 306.0
        series = pd.Series(vals, index=pd.date_range("2023-01-01", periods=14, freq="ME"))
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = series
        with patch("aer.data.macro._get_fred", return_value=mock_fred):
            from aer.data.macro import get_macro_snapshot

            snap = get_macro_snapshot()
        assert snap["cpi_yoy_pct"] == pytest.approx(2.0, abs=0.01)

    def test_unavailable_series_yields_none(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)

        def side_effect(series_id: str, *args, **kwargs) -> pd.Series:
            if series_id == "VIXCLS":
                raise Exception("unavailable")
            return pd.Series([5.0], index=pd.to_datetime(["2024-03-01"]))

        mock_fred = MagicMock()
        mock_fred.get_series.side_effect = side_effect
        with patch("aer.data.macro._get_fred", return_value=mock_fred):
            from aer.data.macro import get_macro_snapshot

            snap = get_macro_snapshot()
        assert snap.get("vix") is None

    def test_second_call_returns_from_cache(self, fresh_cache, monkeypatch) -> None:
        _patch_cache(monkeypatch, fresh_cache)
        call_count = {"n": 0}
        real_fred = _mock_fred(5.0, 14)

        def counted_get(key: str) -> object:
            if key == "macro:snapshot":
                call_count["n"] += 1
            return fresh_cache.get(key)

        with patch("aer.data.macro._get_fred", return_value=real_fred):
            from aer.data.macro import get_macro_snapshot

            snap1 = get_macro_snapshot()
            snap2 = get_macro_snapshot()
        assert snap1["fed_funds_rate"] == snap2["fed_funds_rate"]
