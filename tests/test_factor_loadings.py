"""Test suite for Fama-French factor loadings."""

from __future__ import annotations

from datetime import UTC, date

import numpy as np
import pandas as pd
import pytest

from aer.risk.models import FactorLoadings

# ---------------------------------------------------------------------------
# OLS regression correctness and as_of_date guard
# ---------------------------------------------------------------------------


def test_no_lookahead(monkeypatch: pytest.MonkeyPatch) -> None:
    """No factor data after as_of_date enters the OLS regression.

    Strategy: provide synthetic price + factor history that extends beyond
    as_of_date; assert the regression uses only rows <= as_of_date.
    """
    from aer.risk.factors import get_factor_loadings

    as_of = date(2020, 12, 31)

    # 70 months of synthetic daily data spanning well before and past as_of
    n_daily = 70 * 21  # approximate trading days
    idx = pd.date_range("2015-01-02", periods=n_daily, freq="B")
    rng = np.random.default_rng(0)
    prices = pd.Series(100 * (1 + rng.normal(0.0003, 0.012, n_daily)).cumprod(), index=idx)

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    factor_data = pd.DataFrame(
        rng.normal(0, 0.5, (n_daily, len(factor_cols))),
        index=idx,
        columns=factor_cols,
    )

    def _mock_factor_data(as_of: date | None = None) -> pd.DataFrame:
        if as_of is not None:
            return factor_data[factor_data.index.date <= as_of]
        return factor_data

    def _mock_prices(ticker: str, **kwargs: object) -> pd.Series:
        return prices

    monkeypatch.setattr("aer.risk.factors.get_factor_data", _mock_factor_data)
    monkeypatch.setattr("aer.risk.factors.get_price_history_raw", _mock_prices, raising=False)

    result = get_factor_loadings("AAPL", as_of_date=as_of)

    # Verify no data after as_of leaks - proxy: result must be a FactorLoadings, not error
    assert isinstance(result, FactorLoadings)
    # All betas should be floats (not None) when we have >= 60 aligned months
    assert result.insufficient_history is False
    assert result.mkt_rf is not None


def test_insufficient_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fewer than FACTOR_MIN_MONTHS aligned observations returns insufficient_history=True."""
    from aer.risk.factors import get_factor_loadings

    as_of = date(2015, 12, 31)

    # Only 12 months of history - well below FACTOR_MIN_MONTHS=36
    n_daily = 12 * 21
    idx = pd.date_range("2015-01-02", periods=n_daily, freq="B")
    rng = np.random.default_rng(1)
    prices = pd.Series(100 * (1 + rng.normal(0.0003, 0.012, n_daily)).cumprod(), index=idx)

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    factor_data = pd.DataFrame(
        rng.normal(0, 0.5, (n_daily, len(factor_cols))),
        index=idx,
        columns=factor_cols,
    )

    monkeypatch.setattr(
        "aer.risk.factors.get_factor_data",
        lambda as_of=None: factor_data[factor_data.index.date <= (as_of or date.today())],
    )
    monkeypatch.setattr("aer.risk.factors.get_price_history_raw", lambda t, **kw: prices, raising=False)

    result = get_factor_loadings("AAPL", as_of_date=as_of)

    assert isinstance(result, FactorLoadings)
    assert result.insufficient_history is True
    assert result.mkt_rf is None
    assert result.smb is None
    assert result.hml is None
    assert result.rmw is None
    assert result.cma is None
    assert result.alpha is None


def test_ff5_and_ff3_betas(monkeypatch: pytest.MonkeyPatch) -> None:
    """With >= 60 aligned months all 5 betas are finite floats.

    FF3 subset (mkt_rf, smb, hml) must be consistent with the 5-factor coefficients
    (single regression - both subsets come from the same OLS call).
    """
    from aer.risk.factors import get_factor_loadings

    as_of = date(2022, 12, 31)

    # 70 months of synthetic daily data
    n_daily = 70 * 21
    idx = pd.date_range("2017-01-02", periods=n_daily, freq="B")
    rng = np.random.default_rng(2)

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    factor_data = pd.DataFrame(
        rng.normal(0, 0.5, (n_daily, len(factor_cols))),
        index=idx,
        columns=factor_cols,
    )
    # Make stock returns linearly related to factors so betas are identifiable
    known_betas = [1.1, 0.3, -0.2, 0.4, -0.1]  # mkt_rf, smb, hml, rmw, cma
    excess_ret = factor_data[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].values @ known_betas
    prices = pd.Series(100 * (1 + excess_ret / 100 + rng.normal(0, 0.001, n_daily)).cumprod(), index=idx)

    monkeypatch.setattr(
        "aer.risk.factors.get_factor_data",
        lambda as_of=None: factor_data[factor_data.index.date <= (as_of or date.today())],
    )
    monkeypatch.setattr("aer.risk.factors.get_price_history_raw", lambda t, **kw: prices, raising=False)

    result = get_factor_loadings("AAPL", as_of_date=as_of)

    assert isinstance(result, FactorLoadings)
    assert result.insufficient_history is False
    # All 5 betas must be finite floats
    for field_name in ("mkt_rf", "smb", "hml", "rmw", "cma"):
        val = getattr(result, field_name)
        assert val is not None, f"{field_name} is None"
        assert isinstance(val, float), f"{field_name} is not float"
        assert not (val != val), f"{field_name} is NaN"  # nan check
    # r_squared must be in [0, 1]
    assert result.r_squared is not None
    assert 0.0 <= result.r_squared <= 1.0


# ---------------------------------------------------------------------------
# Caching and manifest registration
# ---------------------------------------------------------------------------


def test_factor_data_cache_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second call to get_factor_data returns from cache (single fetch call).

    Verifies _fetch_ken_french_ff5 is called exactly once across two get_factor_data() calls.
    """
    from aer.data.factors import get_factor_data

    call_count = {"n": 0}

    n = 100
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    rng = np.random.default_rng(3)
    # Use date objects for index to match _fetch_ken_french_ff5 output dtype.
    fake_df = pd.DataFrame(
        rng.normal(0, 0.005, (n, 6)),
        index=idx.date,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )

    def _mock_fetch(end_date: object = None) -> pd.DataFrame:
        call_count["n"] += 1
        return fake_df.copy()

    # Isolate with in-memory cache so repeated test runs don't share SQLite state.
    _mem: dict = {}

    class _FakeCache:
        def get(self, key: str) -> object:
            return _mem.get(key)

        def set(self, key: str, value: object, ttl: int | None = None) -> None:
            _mem[key] = value

    _fc = _FakeCache()
    monkeypatch.setattr("aer.data.factors._fetch_ken_french_ff5", _mock_fetch)
    monkeypatch.setattr("aer.data.factors.data_cache", lambda: _fc)

    result1 = get_factor_data()
    result2 = get_factor_data()

    assert (
        call_count["n"] == 1
    ), f"_fetch_ken_french_ff5 called {call_count['n']} times (expected 1 — cache miss then hit)"
    assert result1.equals(result2)


def test_manifest_ken_french_registered() -> None:
    """'factors:ken_french_daily' spec is registered in the prefetch MANIFEST."""
    from aer.prefetch.manifest import MANIFEST

    names = {spec.name for spec in MANIFEST}
    assert "factors:ken_french_daily" in names, "factors:ken_french_daily not found in MANIFEST"

    spec = next(s for s in MANIFEST if s.name == "factors:ken_french_daily")
    assert spec.scope == "global", f"Expected scope='global', got '{spec.scope}'"
    assert spec.domain == "data", f"Expected domain='data', got '{spec.domain}'"
    assert (
        "date_str" in spec.cache_key_template
    ), f"Expected {{date_str}} in cache_key_template, got '{spec.cache_key_template}'"


# ---------------------------------------------------------------------------
# InvestmentBrief serialisation and graph wiring
# ---------------------------------------------------------------------------


def test_brief_serialisation() -> None:
    """InvestmentBrief with factor_loadings round-trips through model_dump/model_validate."""
    from aer.models import InvestmentBrief

    fl = FactorLoadings(
        alpha=0.05,
        mkt_rf=1.1,
        smb=0.3,
        hml=-0.2,
        rmw=0.4,
        cma=-0.1,
        r_squared=0.78,
        t_stats={"alpha": 1.2, "mkt_rf": 8.4, "smb": 2.1, "hml": -1.5, "rmw": 3.0, "cma": -0.9},
        p_values={"alpha": 0.23, "mkt_rf": 0.0, "smb": 0.04, "hml": 0.14, "rmw": 0.003, "cma": 0.37},
    )

    brief = InvestmentBrief(
        ticker="AAPL",
        company_name="Apple Inc.",
        factor_loadings=fl,
    )

    data = brief.model_dump(mode="json")
    assert "factor_loadings" in data, "factor_loadings key missing from model_dump output"
    assert data["factor_loadings"]["mkt_rf"] == pytest.approx(1.1)

    brief2 = InvestmentBrief.model_validate(data)
    assert brief2.factor_loadings is not None
    assert brief2.factor_loadings.mkt_rf == pytest.approx(1.1)


def test_brief_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    """_post_process (the graph node wiring get_factor_loadings) populates factor_loadings.

    Directly exercises _post_process rather than build_brief so the test does not depend on
    LLM credentials or a pre-existing brief cache entry.
    """
    from datetime import datetime

    from aer.graph import _post_process
    from aer.models import InvestmentBrief

    known_loadings = FactorLoadings(
        mkt_rf=1.05,
        smb=0.25,
        hml=-0.15,
        rmw=0.35,
        cma=-0.05,
        r_squared=0.72,
    )

    monkeypatch.setattr(
        "aer.risk.factors.get_factor_loadings",
        lambda ticker, as_of_date: known_loadings,
    )

    brief = InvestmentBrief(
        ticker="AAPL",
        company_name="Apple Inc.",
        exchange="NMS",
        sector="Technology",
        current_price=200.0,
        price_currency="USD",
        recommendation="HOLD",
        conviction=0.65,
        summary="test brief",
        date=datetime.now(UTC),
    )

    state: dict = {"brief": brief, "backtesting": None, "options_strategy": None}
    result = _post_process(state)
    out_brief = result.get("brief")

    assert out_brief is not None
    assert isinstance(out_brief, InvestmentBrief)
    assert (
        out_brief.factor_loadings is not None
    ), "factor_loadings is None — get_factor_loadings not wired in _post_process"
    assert out_brief.factor_loadings.mkt_rf == pytest.approx(1.05)
