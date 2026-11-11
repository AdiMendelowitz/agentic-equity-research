"""Tests for positioning agent: build_positioning_prompt, confidence formula, run()."""

from unittest.mock import MagicMock, patch

import pytest

from aer.data.cache import TTL_ANALYSIS, Cache
from aer.utils.cache import _DomainCache


@pytest.fixture
def fresh_cache(tmp_path) -> Cache:
    return Cache(db_path=str(tmp_path / "positioning.db"))


@pytest.fixture
def fresh_domain_cache(fresh_cache: Cache) -> _DomainCache:
    """_DomainCache wrapping fresh_cache with agent domain — mirrors agent_cache() singleton."""
    return _DomainCache(fresh_cache, TTL_ANALYSIS, "agent")


def _flow(available: bool = True) -> dict:
    if not available:
        return {"ticker": "AAPL", "available": False}
    return {
        "ticker": "AAPL",
        "available": True,
        "current_price": 185.5,
        "total_call_volume": 50000,
        "total_put_volume": 40000,
        "total_call_oi": 200000,
        "total_put_oi": 180000,
        "pc_volume_ratio": 0.80,
        "pc_oi_ratio": 0.90,
        "atm_iv_call": 0.22,
        "atm_iv_put": 0.25,
        "iv_skew": 0.03,
        "expirations": [],
    }


def _own() -> dict:
    return {
        "ticker": "AAPL",
        "insider_pct": 0.07,
        "institution_pct": 0.62,
        "short_float_pct": 0.008,
        "short_ratio": 1.5,
        "institutional_holders": [
            {"holder": "Vanguard", "shares": 1_000_000, "pct_held": 0.08},
            {"holder": "BlackRock", "shares": 900_000, "pct_held": 0.07},
        ],
        "insider_transactions": [],
        "insider_buys_recent": 3,
        "insider_sells_recent": 1,
        "insider_buy_sell_ratio": 3.0,
    }


def _earn() -> dict:
    return {
        "ticker": "AAPL",
        "next_earnings_date": "2026-07-30",
        "eps_estimate": 1.38,
        "eps_low": 1.25,
        "eps_high": 1.52,
        "revenue_estimate": 94_000_000_000.0,
        "earnings_history": [
            {"date": "2026-01-30", "reported_eps": 1.48, "estimated_eps": 1.42, "surprise_pct": 4.2},
            {"date": "2025-10-31", "reported_eps": 1.32, "estimated_eps": 1.30, "surprise_pct": 1.5},
        ],
        "earnings_beat_rate": 0.75,
        "avg_surprise_pct": 3.5,
        "analyst_recommendations": [],
    }


def _breadth() -> dict:
    return {
        "ad_line_slope": 12.5,
        "ad_line_trend": "rising",
        "mcclellan_oscillator": 45.0,
        "mcclellan_signal": "neutral",
        "net_advances_latest": 800,
        "sector_dispersion": 0.032,
        "breadth_score": 0.75,
    }


def _cboe_pc() -> dict:
    return {
        "equity_pc_ratio": 0.68,
        "total_pc_ratio": 0.82,
        "equity_pc_5d_avg": 0.71,
        "sentiment_label": "NEUTRAL",
    }


class TestBuildPrompt:
    def test_ticker_in_prompt(self) -> None:
        from aer.utils.prompt_builders import build_positioning_prompt

        _, prompt = build_positioning_prompt("AAPL", _flow(), _own(), _earn(), _breadth(), _cboe_pc(), "NEUTRAL")
        assert "AAPL" in prompt

    def test_bullish_lean_when_pc_below_one(self) -> None:
        from aer.utils.prompt_builders import build_positioning_prompt

        flow = _flow()
        flow["pc_volume_ratio"] = 0.80
        _, prompt = build_positioning_prompt("AAPL", flow, _own(), _earn(), _breadth(), _cboe_pc(), "NEUTRAL")
        assert "bullish" in prompt

    def test_bearish_lean_when_pc_above_one(self) -> None:
        from aer.utils.prompt_builders import build_positioning_prompt

        flow = _flow()
        flow["pc_volume_ratio"] = 1.5
        _, prompt = build_positioning_prompt("AAPL", flow, _own(), _earn(), _breadth(), _cboe_pc(), "NEUTRAL")
        assert "bearish" in prompt

    def test_put_premium_when_iv_skew_positive(self) -> None:
        from aer.utils.prompt_builders import build_positioning_prompt

        flow = _flow()
        flow["iv_skew"] = 0.05
        _, prompt = build_positioning_prompt("AAPL", flow, _own(), _earn(), _breadth(), _cboe_pc(), "NEUTRAL")
        assert "put premium" in prompt

    def test_call_premium_when_iv_skew_negative(self) -> None:
        from aer.utils.prompt_builders import build_positioning_prompt

        flow = _flow()
        flow["iv_skew"] = -0.05
        _, prompt = build_positioning_prompt("AAPL", flow, _own(), _earn(), _breadth(), _cboe_pc(), "NEUTRAL")
        assert "call premium" in prompt

    def test_options_unavailable_shows_message(self) -> None:
        from aer.utils.prompt_builders import build_positioning_prompt

        _, prompt = build_positioning_prompt(
            "AAPL", _flow(available=False), _own(), _earn(), _breadth(), _cboe_pc(), "NEUTRAL"
        )
        assert "unavailable" in prompt

    def test_next_earnings_included(self) -> None:
        from aer.utils.prompt_builders import build_positioning_prompt

        _, prompt = build_positioning_prompt("AAPL", _flow(), _own(), _earn(), _breadth(), _cboe_pc(), "NEUTRAL")
        assert "2026-07-30" in prompt

    def test_institutional_holders_listed(self) -> None:
        from aer.utils.prompt_builders import build_positioning_prompt

        _, prompt = build_positioning_prompt("AAPL", _flow(), _own(), _earn(), _breadth(), _cboe_pc(), "NEUTRAL")
        assert "Vanguard" in prompt

    def test_breadth_section_included(self) -> None:
        from aer.utils.prompt_builders import build_positioning_prompt

        _, prompt = build_positioning_prompt("AAPL", _flow(), _own(), _earn(), _breadth(), _cboe_pc(), "NEUTRAL")
        assert "McClellan" in prompt
        assert "CBOE" in prompt


def _patch_run(
    domain_cache: _DomainCache,
    ticker: str = "AAPL",
    flow: dict | None = None,
    own: dict | None = None,
    earn: dict | None = None,
    breadth: dict | None = None,
    cboe_pc: dict | None = None,
) -> object:
    mock_response = MagicMock()
    mock_response.content = "Strong positioning."
    with patch("aer.agents.positioning.agent_cache", return_value=domain_cache):
        with patch("aer.agents.positioning.get_options_flow", return_value=flow or _flow()):
            with patch("aer.agents.positioning.get_ownership", return_value=own or _own()):
                with patch("aer.agents.positioning.get_earnings_data", return_value=earn or _earn()):
                    with patch("aer.agents.positioning.get_breadth_snapshot", return_value=breadth or _breadth()):
                        with patch("aer.agents.positioning.get_cboe_pc_ratio", return_value=cboe_pc or _cboe_pc()):
                            with patch("aer.agents.positioning.complete") as mock_complete:
                                mock_complete.return_value = mock_response
                                from aer.agents.positioning import run

                                return run(ticker)


class TestPositioningConfidence:
    def test_all_sources_active(self, fresh_domain_cache: _DomainCache) -> None:

        result = _patch_run(fresh_domain_cache)
        # options available + holders + history + breadth_score + cboe_pc = 5/5
        expected = round(0.2 + (5 / 5) * 0.65, 2)
        assert result.confidence == pytest.approx(expected, abs=0.01)

    def test_no_sources_active(self, fresh_domain_cache: _DomainCache) -> None:
        empty_own = {"ticker": "AAPL", "institutional_holders": []}
        empty_earn = {"ticker": "AAPL", "earnings_history": []}
        empty_breadth = {"breadth_score": None}
        empty_cboe = {"equity_pc_ratio": None}
        result = _patch_run(
            fresh_domain_cache,
            flow=_flow(available=False),
            own=empty_own,
            earn=empty_earn,
            breadth=empty_breadth,
            cboe_pc=empty_cboe,
        )
        expected = round(0.2 + 0.0, 2)
        assert result.confidence == pytest.approx(expected, abs=0.01)


class TestPositioningRun:
    def test_returns_agent_output(self, fresh_domain_cache: _DomainCache) -> None:
        from aer.models import AgentOutput

        result = _patch_run(fresh_domain_cache)
        assert isinstance(result, AgentOutput)
        assert result.agent_name == "positioning"

    def test_sources_include_all(self, fresh_domain_cache: _DomainCache) -> None:
        result = _patch_run(fresh_domain_cache)
        assert "yfinance:options" in result.sources
        assert "yfinance:institutional_holders" in result.sources
        assert "yfinance:earnings" in result.sources
        assert "yfinance:PCCE" in result.sources

    def test_ticker_uppercased(self, fresh_domain_cache: _DomainCache) -> None:
        mock_response = MagicMock()
        mock_response.content = "text"
        with patch("aer.agents.positioning.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.positioning.get_options_flow", return_value=_flow()) as mock_flow:
                with patch("aer.agents.positioning.get_ownership", return_value=_own()):
                    with patch("aer.agents.positioning.get_earnings_data", return_value=_earn()):
                        with patch("aer.agents.positioning.get_breadth_snapshot", return_value=_breadth()):
                            with patch("aer.agents.positioning.get_cboe_pc_ratio", return_value=_cboe_pc()):
                                with patch("aer.agents.positioning.complete") as mock_complete:
                                    mock_complete.return_value = mock_response
                                    from aer.agents.positioning import run

                                    run("aapl")
        mock_flow.assert_called_once_with("AAPL")

    def test_cache_hit_skips_llm(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        from aer.models import AgentOutput

        cached = AgentOutput(agent_name="positioning", content="cached positioning", confidence=0.6)
        fresh_domain_cache.set("agent:positioning:AAPL", cached.model_dump(mode="json"), ttl=3600)
        with patch("aer.agents.positioning.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.positioning.complete") as mock_complete:
                from aer.agents.positioning import run

                result = run("AAPL")
        mock_complete.assert_not_called()
        assert result.content == "cached positioning"

    def test_metadata_key_fields(self, fresh_domain_cache: _DomainCache) -> None:
        result = _patch_run(fresh_domain_cache)
        assert result.metadata["ticker"] == "AAPL"
        assert "pc_volume_ratio" in result.metadata
        assert "breadth_score" in result.metadata
        assert "cboe_equity_pc" in result.metadata

    def test_result_cached(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        _patch_run(fresh_domain_cache)
        assert fresh_cache.get("agent:positioning:AAPL") is not None

    def test_uses_analysis_tier(self, fresh_domain_cache: _DomainCache) -> None:
        mock_response = MagicMock()
        mock_response.content = "text"
        with patch("aer.agents.positioning.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.positioning.get_options_flow", return_value=_flow()):
                with patch("aer.agents.positioning.get_ownership", return_value=_own()):
                    with patch("aer.agents.positioning.get_earnings_data", return_value=_earn()):
                        with patch("aer.agents.positioning.get_breadth_snapshot", return_value=_breadth()):
                            with patch("aer.agents.positioning.get_cboe_pc_ratio", return_value=_cboe_pc()):
                                with patch("aer.agents.positioning.complete") as mock_complete:
                                    mock_complete.return_value = mock_response
                                    from aer.agents.positioning import run

                                    run("AAPL")
        assert mock_complete.call_args.kwargs["tier"] == "ANALYSIS"
