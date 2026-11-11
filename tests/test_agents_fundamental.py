"""Tests for fundamental analysis agent: build_fundamental_prompt, run()."""

from unittest.mock import MagicMock, patch

import pytest

from aer.data.cache import TTL_ANALYSIS, Cache
from aer.utils.cache import _DomainCache


@pytest.fixture
def fresh_cache(tmp_path) -> Cache:
    return Cache(db_path=str(tmp_path / "fund.db"))


@pytest.fixture
def fresh_agent_cache(fresh_cache: Cache) -> _DomainCache:
    return _DomainCache(fresh_cache, TTL_ANALYSIS, "agent")


def _info() -> dict:
    return {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "exchange": "NASDAQ",
        "currency": "USD",
        "market_cap": 3_000_000_000_000,
        "employees": 164000,
        "description": "Apple designs and sells consumer electronics.",
        "website": "https://apple.com",
        "country": "US",
        "pe_ratio": 28.5,
        "forward_pe": 25.0,
        "price_to_book": 45.0,
        "dividend_yield": 0.005,
        "52w_high": 199.62,
        "52w_low": 124.17,
        "beta": 1.21,
        "analyst_target": 210.0,
        "recommendation": "buy",
        "analyst_consensus": {},
    }


def _ratios() -> dict:
    return {
        "revenue_growth_yoy": 8.5,
        "gross_margin": 44.0,
        "operating_margin": 29.0,
        "net_margin": 25.0,
        "fcf_on_equity": 30.0,
        "debt_to_equity": 1.5,
        "net_cash": 50_000_000_000,
        "eps_history": [
            {"period": "2023", "value": 5.89},
            {"period": "2022", "value": 6.11},
        ],
    }


class TestBuildPrompt:
    def test_contains_ticker(self) -> None:
        from aer.utils.prompt_builders import build_fundamental_prompt

        _, prompt = build_fundamental_prompt("AAPL", _info(), 185.5, _ratios())
        assert "AAPL" in prompt

    def test_contains_company_name(self) -> None:
        from aer.utils.prompt_builders import build_fundamental_prompt

        _, prompt = build_fundamental_prompt("AAPL", _info(), 185.5, _ratios())
        assert "Apple Inc." in prompt

    def test_contains_formatted_price(self) -> None:
        from aer.utils.prompt_builders import build_fundamental_prompt

        _, prompt = build_fundamental_prompt("AAPL", _info(), 185.5, _ratios())
        assert "$185.50" in prompt

    def test_none_price_shows_na(self) -> None:
        from aer.utils.prompt_builders import build_fundamental_prompt

        _, prompt = build_fundamental_prompt("AAPL", _info(), None, _ratios())
        assert "N/A" in prompt

    def test_eps_history_rendered(self) -> None:
        from aer.utils.prompt_builders import build_fundamental_prompt

        _, prompt = build_fundamental_prompt("AAPL", _info(), 185.5, _ratios())
        assert "$5.89" in prompt

    def test_sector_included(self) -> None:
        from aer.utils.prompt_builders import build_fundamental_prompt

        _, prompt = build_fundamental_prompt("AAPL", _info(), 185.5, _ratios())
        assert "Technology" in prompt

    def test_recommendation_uppercased(self) -> None:
        from aer.utils.prompt_builders import build_fundamental_prompt

        _, prompt = build_fundamental_prompt("AAPL", _info(), 185.5, _ratios())
        assert "BUY" in prompt


class TestFundamentalRun:
    def _run(self, agent_cache: _DomainCache, ticker: str = "AAPL") -> object:
        mock_response = MagicMock()
        mock_response.content = "Strong fundamentals."
        with patch("aer.agents.fundamental.agent_cache", return_value=agent_cache):
            with patch("aer.agents.fundamental.get_company_info", return_value=_info()):
                with patch("aer.agents.fundamental.get_current_price", return_value=185.5):
                    with patch("aer.agents.fundamental.get_key_ratios", return_value=_ratios()):
                        with patch("aer.agents.fundamental.complete") as mock_complete:
                            mock_complete.return_value = mock_response
                            from aer.agents.fundamental import run

                            return run(ticker)

    def test_returns_agent_output(self, fresh_agent_cache: _DomainCache) -> None:
        from aer.models import AgentOutput

        result = self._run(fresh_agent_cache)
        assert isinstance(result, AgentOutput)
        assert result.agent_name == "fundamental"

    def test_ticker_uppercased(self, fresh_cache: Cache, fresh_agent_cache: _DomainCache) -> None:
        mock_response = MagicMock()
        mock_response.content = "text"
        with patch("aer.agents.fundamental.agent_cache", return_value=fresh_agent_cache):
            with patch("aer.agents.fundamental.get_company_info", return_value=_info()) as mock_info:
                with patch("aer.agents.fundamental.get_current_price", return_value=185.5):
                    with patch("aer.agents.fundamental.get_key_ratios", return_value=_ratios()):
                        with patch("aer.agents.fundamental.complete") as mock_complete:
                            mock_complete.return_value = mock_response
                            from aer.agents.fundamental import run

                            run("aapl")
        mock_info.assert_called_once_with("AAPL")

    def test_cache_hit_skips_llm(self, fresh_cache: Cache, fresh_agent_cache: _DomainCache) -> None:
        from aer.models import AgentOutput

        cached = AgentOutput(agent_name="fundamental", content="cached", confidence=0.8)
        fresh_agent_cache.set("agent:fundamental:AAPL", cached.model_dump(mode="json"), ttl=3600)
        with patch("aer.agents.fundamental.agent_cache", return_value=fresh_agent_cache):
            with patch("aer.agents.fundamental.complete") as mock_complete:
                from aer.agents.fundamental import run

                result = run("AAPL")
        mock_complete.assert_not_called()
        assert result.content == "cached"

    def test_confidence_capped_at_0_9(self, fresh_agent_cache: _DomainCache) -> None:
        result = self._run(fresh_agent_cache)
        assert result.confidence <= 0.9

    def test_confidence_in_0_1_range(self, fresh_agent_cache: _DomainCache) -> None:
        result = self._run(fresh_agent_cache)
        assert 0.0 <= result.confidence <= 1.0

    def test_source_is_yfinance(self, fresh_agent_cache: _DomainCache) -> None:
        result = self._run(fresh_agent_cache)
        assert result.sources == ["yfinance:AAPL"]

    def test_metadata_ticker_price_sector(self, fresh_agent_cache: _DomainCache) -> None:
        result = self._run(fresh_agent_cache)
        assert result.metadata["ticker"] == "AAPL"
        assert result.metadata["price"] == pytest.approx(185.5)
        assert result.metadata["sector"] == "Technology"

    def test_result_cached(self, fresh_cache: Cache, fresh_agent_cache: _DomainCache) -> None:
        self._run(fresh_agent_cache)
        assert fresh_cache.get("agent:fundamental:AAPL") is not None

    def test_uses_analysis_tier(self, fresh_agent_cache: _DomainCache) -> None:
        mock_response = MagicMock()
        mock_response.content = "text"
        with patch("aer.agents.fundamental.agent_cache", return_value=fresh_agent_cache):
            with patch("aer.agents.fundamental.get_company_info", return_value=_info()):
                with patch("aer.agents.fundamental.get_current_price", return_value=185.5):
                    with patch("aer.agents.fundamental.get_key_ratios", return_value=_ratios()):
                        with patch("aer.agents.fundamental.complete") as mock_complete:
                            mock_complete.return_value = mock_response
                            from aer.agents.fundamental import run

                            run("AAPL")
        assert mock_complete.call_args.kwargs["tier"] == "ANALYSIS"
