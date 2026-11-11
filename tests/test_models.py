"""Tests for Pydantic data models: AgentOutput and InvestmentBrief."""

import pytest
from pydantic import ValidationError

from aer.models import AgentOutput, InvestmentBrief


class TestAgentOutput:
    def test_minimal_construction(self) -> None:
        out = AgentOutput(agent_name="test", content="analysis text")
        assert out.agent_name == "test"
        assert out.content == "analysis text"

    def test_missing_agent_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentOutput(content="text")  # type: ignore[call-arg]

    def test_missing_content_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentOutput(agent_name="test")  # type: ignore[call-arg]

    def test_confidence_lower_bound(self) -> None:
        AgentOutput(agent_name="a", content="b", confidence=0.0)

    def test_confidence_upper_bound(self) -> None:
        AgentOutput(agent_name="a", content="b", confidence=1.0)

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentOutput(agent_name="a", content="b", confidence=-0.001)

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentOutput(agent_name="a", content="b", confidence=1.001)

    def test_default_confidence_is_0_5(self) -> None:
        assert AgentOutput(agent_name="a", content="b").confidence == 0.5

    def test_timestamp_has_timezone(self) -> None:
        out = AgentOutput(agent_name="a", content="b")
        assert out.timestamp.tzinfo is not None

    def test_sources_defaults_empty(self) -> None:
        assert AgentOutput(agent_name="a", content="b").sources == []

    def test_metadata_defaults_empty(self) -> None:
        assert AgentOutput(agent_name="a", content="b").metadata == {}

    def test_sources_not_shared_between_instances(self) -> None:
        a = AgentOutput(agent_name="a", content="b")
        b = AgentOutput(agent_name="a", content="b")
        a.sources.append("x")
        assert "x" not in b.sources

    def test_json_round_trip(self) -> None:
        out = AgentOutput(
            agent_name="fundamental",
            content="strong earnings",
            confidence=0.8,
            sources=["yfinance:AAPL"],
            metadata={"price": 185.5},
        )
        data = out.model_dump(mode="json")
        restored = AgentOutput(**data)
        assert restored.agent_name == out.agent_name
        assert restored.confidence == out.confidence
        assert restored.sources == out.sources


class TestInvestmentBrief:
    def test_minimal_construction(self) -> None:
        brief = InvestmentBrief(ticker="AAPL", company_name="Apple Inc.")
        assert brief.ticker == "AAPL"
        assert brief.company_name == "Apple Inc."

    def test_missing_ticker_raises(self) -> None:
        with pytest.raises(ValidationError):
            InvestmentBrief(company_name="Apple")  # type: ignore[call-arg]

    def test_missing_company_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            InvestmentBrief(ticker="AAPL")  # type: ignore[call-arg]

    @pytest.mark.parametrize("rec", ["BUY", "HOLD", "SELL", "AVOID"])
    def test_valid_recommendations(self, rec: str) -> None:
        b = InvestmentBrief(ticker="T", company_name="Test", recommendation=rec)
        assert b.recommendation == rec

    def test_invalid_recommendation_raises(self) -> None:
        with pytest.raises(ValidationError):
            InvestmentBrief(ticker="T", company_name="Test", recommendation="STRONG_BUY")

    def test_conviction_lower_bound(self) -> None:
        InvestmentBrief(ticker="T", company_name="Test", conviction=0.0)

    def test_conviction_upper_bound(self) -> None:
        InvestmentBrief(ticker="T", company_name="Test", conviction=1.0)

    def test_conviction_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            InvestmentBrief(ticker="T", company_name="Test", conviction=-0.01)

    def test_conviction_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            InvestmentBrief(ticker="T", company_name="Test", conviction=1.01)

    @pytest.mark.parametrize("th", ["short", "medium", "long"])
    def test_valid_time_horizons(self, th: str) -> None:
        b = InvestmentBrief(ticker="T", company_name="Test", time_horizon=th)
        assert b.time_horizon == th

    def test_invalid_time_horizon_raises(self) -> None:
        with pytest.raises(ValidationError):
            InvestmentBrief(ticker="T", company_name="Test", time_horizon="very_long")

    def test_agent_outputs_default_none(self) -> None:
        brief = InvestmentBrief(ticker="T", company_name="Test")
        for field in ("fundamental", "technical", "sentiment", "macro", "positioning"):
            assert getattr(brief, field) is None

    def test_agent_output_attached(self) -> None:
        out = AgentOutput(agent_name="fundamental", content="analysis")
        brief = InvestmentBrief(ticker="T", company_name="Test", fundamental=out)
        assert brief.fundamental is not None
        assert brief.fundamental.agent_name == "fundamental"

    def test_scalar_defaults(self) -> None:
        brief = InvestmentBrief(ticker="T", company_name="Test")
        assert brief.recommendation == "HOLD"
        assert brief.conviction == 0.5
        assert brief.time_horizon == "medium"
        assert brief.exchange == ""
        assert brief.sector == ""
        assert brief.price_currency == "USD"
        assert brief.current_price is None
        assert brief.price_target is None
        assert brief.risks == []
        assert brief.catalysts == []
        assert brief.summary == ""

    def test_json_round_trip(self) -> None:
        brief = InvestmentBrief(
            ticker="AAPL",
            company_name="Apple",
            recommendation="BUY",
            conviction=0.8,
            risks=["rate risk"],
            catalysts=["AI"],
        )
        restored = InvestmentBrief(**brief.model_dump(mode="json"))
        assert restored.ticker == "AAPL"
        assert restored.recommendation == "BUY"
        assert restored.conviction == 0.8
        assert restored.risks == ["rate risk"]
