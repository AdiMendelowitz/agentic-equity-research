"""Tests for macro context agent: build_macro_prompt, run(), confidence formula."""

from unittest.mock import MagicMock, patch

import pytest

from aer.data.cache import TTL_ANALYSIS, Cache
from aer.utils.cache import _DomainCache


@pytest.fixture
def fresh_cache(tmp_path) -> Cache:
    return Cache(db_path=str(tmp_path / "macro_agent.db"))


@pytest.fixture
def fresh_domain_cache(fresh_cache: Cache) -> _DomainCache:
    """_DomainCache wrapping fresh_cache with agent domain — mirrors agent_cache() singleton."""
    return _DomainCache(fresh_cache, TTL_ANALYSIS, "agent")


def _snap(all_none: bool = False) -> dict:
    if all_none:
        keys = (
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
            "hy_spread",
            "ig_spread",
            "real_yield_10y",
            "breakeven_10y",
            "net_liquidity_bn",
            "net_liquidity_trend",
            "dxy",
            "dxy_change_pct",
            "usdjpy",
            "jobless_claims",
            "consumer_sentiment",
        )
        return {k: None for k in keys}
    return {
        "fed_funds_rate": 5.33,
        "unemployment": 3.7,
        "treasury_10y": 4.25,
        "treasury_2y": 4.88,
        "yield_curve": -0.63,
        "gdp": 28500.0,
        "pce": 119.5,
        "vix": 13.5,
        "sp500": 5200.0,
        "cpi": 314.0,
        "cpi_yoy_pct": 3.2,
        "hy_spread": 320.0,
        "ig_spread": 95.0,
        "bbb_spread": 145.0,
        "real_yield_10y": 2.1,
        "breakeven_10y": 2.3,
        "breakeven_5y5y": 2.4,
        "net_liquidity_bn": 5800.0,
        "net_liquidity_trend": "neutral",
        "dxy": 104.5,
        "dxy_change_pct": 0.12,
        "eurusd": 1.082,
        "usdjpy": 149.5,
        "jobless_claims": 215.0,
        "consumer_sentiment": 78.5,
        "retail_sales": 720.0,
    }


def _regime() -> dict:
    return {
        "regime": "BULL_STEADY",
        "description": "Risk-on expansion with low volatility.",
        "equity_bias": "BULLISH",
        "score": 2,
        "signals": ["VIX 13.5 — low fear (bullish)", "HY spread 320bps — tight (bullish)"],
    }


class TestBuildPrompt:
    def test_contains_ticker_and_sector(self) -> None:
        from aer.utils.prompt_builders import build_macro_prompt

        _, prompt = build_macro_prompt("AAPL", "Technology", _snap(), _regime(), "FAVORABLE")
        assert "AAPL" in prompt
        assert "Technology" in prompt

    def test_fed_funds_rate_formatted(self) -> None:
        from aer.utils.prompt_builders import build_macro_prompt

        _, prompt = build_macro_prompt("AAPL", "Technology", _snap(), _regime(), "FAVORABLE")
        assert "5.33%" in prompt

    def test_none_values_show_na(self) -> None:
        from aer.utils.prompt_builders import build_macro_prompt

        _, prompt = build_macro_prompt("AAPL", "Technology", _snap(all_none=True), _regime(), "FAVORABLE")
        assert "N/A" in prompt

    def test_contains_fred_fields(self) -> None:
        from aer.utils.prompt_builders import build_macro_prompt

        _, prompt = build_macro_prompt("AAPL", "Technology", _snap(), _regime(), "FAVORABLE")
        for label in ("Fed Funds Rate", "Unemployment", "10Y Treasury", "VIX"):
            assert label in prompt

    def test_contains_credit_spread_fields(self) -> None:
        from aer.utils.prompt_builders import build_macro_prompt

        _, prompt = build_macro_prompt("AAPL", "Technology", _snap(), _regime(), "FAVORABLE")
        assert "HY OAS" in prompt
        assert "IG OAS" in prompt

    def test_contains_regime_classification(self) -> None:
        from aer.utils.prompt_builders import build_macro_prompt

        _, prompt = build_macro_prompt("AAPL", "Technology", _snap(), _regime(), "FAVORABLE")
        assert "BULL_STEADY" in prompt
        assert "BULLISH" in prompt


class TestMacroConfidence:
    def test_all_series_available(self, fresh_domain_cache: _DomainCache) -> None:
        mock_response = MagicMock()
        mock_response.content = "text"
        snap = _snap()
        available = sum(1 for v in snap.values() if v is not None)
        expected = round(min(0.9, 0.5 + (available / max(len(snap), 1)) * 0.4), 2)
        with patch("aer.agents.macro.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.macro.get_company_metadata", return_value={"sector": "Tech"}):
                with patch("aer.agents.macro.get_macro_snapshot", return_value=snap):
                    with patch("aer.agents.macro.classify_regime", return_value=_regime()):
                        with patch("aer.agents.macro.complete") as mock_complete:
                            mock_complete.return_value = mock_response
                            from aer.agents.macro import run

                            result = run("AAPL")
        assert result.confidence == pytest.approx(expected, abs=0.01)

    def test_no_series_available_gives_0_5_base(self, fresh_domain_cache: _DomainCache) -> None:
        mock_response = MagicMock()
        mock_response.content = "text"
        snap = _snap(all_none=True)
        with patch("aer.agents.macro.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.macro.get_company_metadata", return_value={"sector": "Tech"}):
                with patch("aer.agents.macro.get_macro_snapshot", return_value=snap):
                    with patch("aer.agents.macro.classify_regime", return_value=_regime()):
                        with patch("aer.agents.macro.complete") as mock_complete:
                            mock_complete.return_value = mock_response
                            from aer.agents.macro import run

                            result = run("AAPL")
        assert result.confidence == pytest.approx(0.5, abs=0.01)

    def test_confidence_never_exceeds_0_9(self, fresh_domain_cache: _DomainCache) -> None:
        mock_response = MagicMock()
        mock_response.content = "text"
        with patch("aer.agents.macro.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.macro.get_company_metadata", return_value={"sector": "Tech"}):
                with patch("aer.agents.macro.get_macro_snapshot", return_value=_snap()):
                    with patch("aer.agents.macro.classify_regime", return_value=_regime()):
                        with patch("aer.agents.macro.complete") as mock_complete:
                            mock_complete.return_value = mock_response
                            from aer.agents.macro import run

                            result = run("AAPL")
        assert result.confidence <= 0.9


class TestMacroRun:
    def _run(self, domain_cache: _DomainCache, ticker: str = "AAPL") -> object:
        mock_response = MagicMock()
        mock_response.content = "Macro regime is restrictive."
        with patch("aer.agents.macro.agent_cache", return_value=domain_cache):
            with patch("aer.agents.macro.get_company_metadata", return_value={"sector": "Technology"}):
                with patch("aer.agents.macro.get_macro_snapshot", return_value=_snap()):
                    with patch("aer.agents.macro.classify_regime", return_value=_regime()):
                        with patch("aer.agents.macro.complete") as mock_complete:
                            mock_complete.return_value = mock_response
                            from aer.agents.macro import run

                            return run(ticker)

    def test_returns_agent_output(self, fresh_domain_cache: _DomainCache) -> None:
        from aer.models import AgentOutput

        result = self._run(fresh_domain_cache)
        assert isinstance(result, AgentOutput)
        assert result.agent_name == "macro"

    def test_sources_are_fred_series(self, fresh_domain_cache: _DomainCache) -> None:
        result = self._run(fresh_domain_cache)
        assert "fredapi:FEDFUNDS" in result.sources
        assert "fredapi:CPIAUCSL" in result.sources
        assert "fredapi:DGS10" in result.sources

    def test_sources_include_credit_and_fx(self, fresh_domain_cache: _DomainCache) -> None:
        result = self._run(fresh_domain_cache)
        assert "fredapi:BAMLH0A0HYM2OAS" in result.sources
        assert "yfinance:DX-Y.NYB" in result.sources

    def test_ticker_uppercased(self, fresh_domain_cache: _DomainCache) -> None:
        mock_response = MagicMock()
        mock_response.content = "text"
        with patch("aer.agents.macro.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.macro.get_company_metadata", return_value={"sector": "Tech"}) as mock_info:
                with patch("aer.agents.macro.get_macro_snapshot", return_value=_snap()):
                    with patch("aer.agents.macro.classify_regime", return_value=_regime()):
                        with patch("aer.agents.macro.complete") as mock_complete:
                            mock_complete.return_value = mock_response
                            from aer.agents.macro import run

                            run("aapl")
        mock_info.assert_called_once_with("AAPL")

    def test_cache_hit_skips_llm(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        from aer.models import AgentOutput

        cached = AgentOutput(agent_name="macro", content="cached macro", confidence=0.7)
        fresh_domain_cache.set("agent:macro:AAPL", cached.model_dump(mode="json"), ttl=3600)
        with patch("aer.agents.macro.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.macro.complete") as mock_complete:
                from aer.agents.macro import run

                result = run("AAPL")
        mock_complete.assert_not_called()
        assert result.content == "cached macro"

    def test_metadata_contains_snapshot_and_regime(self, fresh_domain_cache: _DomainCache) -> None:
        result = self._run(fresh_domain_cache)
        assert "macro_snapshot" in result.metadata
        assert "regime" in result.metadata
        assert result.metadata["sector"] == "Technology"
        assert result.metadata["ticker"] == "AAPL"

    def test_result_cached(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        self._run(fresh_domain_cache)
        assert fresh_cache.get("agent:macro:AAPL") is not None

    def test_uses_analysis_tier(self, fresh_domain_cache: _DomainCache) -> None:
        mock_response = MagicMock()
        mock_response.content = "text"
        with patch("aer.agents.macro.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.macro.get_company_metadata", return_value={"sector": "Technology"}):
                with patch("aer.agents.macro.get_macro_snapshot", return_value=_snap()):
                    with patch("aer.agents.macro.classify_regime", return_value=_regime()):
                        with patch("aer.agents.macro.complete") as mock_complete:
                            mock_complete.return_value = mock_response
                            from aer.agents.macro import run

                            run("AAPL")
        assert mock_complete.call_args.kwargs["tier"] == "ANALYSIS"
