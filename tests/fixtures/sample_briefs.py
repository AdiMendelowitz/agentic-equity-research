"""Test fixtures for InvestmentBrief objects with varying signal patterns."""

from datetime import UTC, datetime, timedelta

from aer.models import AgentOutput, InvestmentBrief


def sample_brief_full() -> InvestmentBrief:
    """InvestmentBrief with all 5 agents present, mixed signals (BUY/HOLD/SELL/AVOID)."""
    return InvestmentBrief(
        ticker="AAPL",
        company_name="Apple Inc.",
        exchange="NASDAQ",
        sector="Technology",
        date=datetime.now(UTC),
        current_price=150.0,
        price_target=170.0,
        recommendation="BUY",
        conviction=0.8,
        summary="Strong fundamentals with positive technical signals.",
        fundamental=AgentOutput(
            agent_name="fundamental",
            content="Earnings growth strong, PE reasonable.",
            metadata={"signal": "BUY"},
            confidence=0.85,
        ),
        technical=AgentOutput(
            agent_name="technical",
            content="Price above 200-day MA, RSI neutral.",
            metadata={"signal": "HOLD"},
            confidence=0.65,
        ),
        sentiment=AgentOutput(
            agent_name="sentiment",
            content="News sentiment mixed, social media bullish.",
            metadata={"signal": "SELL"},
            confidence=0.55,
        ),
        macro=AgentOutput(
            agent_name="macro",
            content="Rate environment challenging for growth.",
            metadata={"signal": "AVOID"},
            confidence=0.60,
        ),
        positioning=AgentOutput(
            agent_name="positioning",
            content="Smart money accumulating, short interest low.",
            metadata={"signal": "BUY"},
            confidence=0.75,
        ),
    )


def sample_brief_partial() -> InvestmentBrief:
    """InvestmentBrief with 2 agents None (fundamental, technical present; sentiment, macro, positioning None)."""
    return InvestmentBrief(
        ticker="MSFT",
        company_name="Microsoft Corporation",
        exchange="NASDAQ",
        sector="Technology",
        date=datetime.now(UTC),
        current_price=400.0,
        price_target=420.0,
        recommendation="HOLD",
        conviction=0.6,
        summary="Solid operations, cautious on valuation.",
        fundamental=AgentOutput(
            agent_name="fundamental",
            content="Cloud segment driving growth.",
            metadata={"signal": "BUY"},
            confidence=0.80,
        ),
        technical=AgentOutput(
            agent_name="technical",
            content="Consolidating near resistance.",
            metadata={"signal": "HOLD"},
            confidence=0.70,
        ),
        sentiment=None,
        macro=None,
        positioning=None,
    )


def sample_brief_empty() -> InvestmentBrief:
    """InvestmentBrief with all agents None (no signals available)."""
    return InvestmentBrief(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        exchange="NASDAQ",
        sector="Technology",
        date=datetime.now(UTC),
        current_price=800.0,
        price_target=900.0,
        recommendation="HOLD",
        conviction=0.5,
        summary="Data unavailable for analysis.",
        fundamental=None,
        technical=None,
        sentiment=None,
        macro=None,
        positioning=None,
    )


def sample_brief_stale(hours_old: float) -> InvestmentBrief:
    """InvestmentBrief with date set to N hours ago for staleness testing."""
    return InvestmentBrief(
        ticker="TSLA",
        company_name="Tesla Inc.",
        exchange="NASDAQ",
        sector="Technology",
        date=datetime.now(UTC) - timedelta(hours=hours_old),
        current_price=250.0,
        price_target=300.0,
        recommendation="BUY",
        conviction=0.7,
        summary="Growth story intact despite recent pullback.",
        fundamental=AgentOutput(
            agent_name="fundamental",
            content="Revenue growth strong, margins expanding.",
            metadata={"signal": "BUY"},
            confidence=0.75,
        ),
        technical=AgentOutput(
            agent_name="technical",
            content="Breakout from consolidation range.",
            metadata={"signal": "BUY"},
            confidence=0.70,
        ),
        sentiment=AgentOutput(
            agent_name="sentiment",
            content="Positive social sentiment, bullish bias.",
            metadata={"signal": "BUY"},
            confidence=0.65,
        ),
        macro=AgentOutput(
            agent_name="macro",
            content="Tech sector benefiting from AI themes.",
            metadata={"signal": "HOLD"},
            confidence=0.60,
        ),
        positioning=AgentOutput(
            agent_name="positioning",
            content="Institutional buying, low short float.",
            metadata={"signal": "BUY"},
            confidence=0.80,
        ),
    )
