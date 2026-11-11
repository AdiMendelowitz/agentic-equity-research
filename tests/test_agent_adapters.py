"""Tests for src/utils/agent_adapters.py — pure unit tests, no I/O, no LLM calls."""

from datetime import UTC, datetime
from unittest.mock import patch

from aer.models import AgentOutput


def _make_agent_output(
    agent_name: str,
    metadata: dict,
    confidence: float = 0.8,
    content: str = "",
) -> AgentOutput:
    return AgentOutput(
        agent_name=agent_name,
        content=content,
        confidence=confidence,
        metadata=metadata,
        timestamp=datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC),
    )


def test_adapt_all_skips_none_agents() -> None:
    from aer.utils.agent_adapters import adapt_all

    result = adapt_all({}, "AAPL", "RISK_ON")
    assert result == []


def test_adapt_all_injects_regime_label() -> None:
    from aer.utils.agent_adapters import adapt_all

    output = _make_agent_output(
        "macro",
        metadata={"regime": {"label": "RISK_ON"}, "ticker": "AAPL"},
    )
    result = adapt_all({"macro": output}, "AAPL", "BULLISH_OVERRIDE")
    assert len(result) == 1
    assert result[0].regime_label == "BULLISH_OVERRIDE"


def test_technical_adapter_bullish_to_buy() -> None:
    from aer.utils.agent_adapters import adapt_all

    output = _make_agent_output(
        "technical",
        metadata={"signals": {"overall": "BULLISH"}, "ticker": "AAPL"},
    )
    result = adapt_all({"technical": output}, "AAPL", "RISK_ON")
    assert len(result) == 1
    assert result[0].view == "BUY"


def test_adapt_all_handles_adapter_exception() -> None:
    from aer.utils.agent_adapters import _ADAPTERS, adapt_all

    output = _make_agent_output(
        "technical",
        metadata={"signals": {"overall": "BULLISH"}, "ticker": "AAPL"},
    )
    with patch.dict(_ADAPTERS, {"technical": lambda _: (_ for _ in ()).throw(RuntimeError("boom"))}):
        result = adapt_all({"technical": output}, "AAPL", "RISK_ON")
    assert result == []
