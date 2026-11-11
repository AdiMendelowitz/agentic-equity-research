"""Positioning agent - options flow, institutional/insider ownership, earnings setup."""

import logging

from aer.data.breadth import get_breadth_snapshot
from aer.data.earnings import get_earnings_data
from aer.data.options_flow import get_options_flow
from aer.data.ownership import get_ownership
from aer.data.sentiment_feeds import get_cboe_pc_ratio
from aer.models import AgentOutput
from aer.utils.cache import agent_cache
from aer.utils.llm_client import complete
from aer.utils.prompt_builders import build_positioning_prompt
from aer.utils.python_replacements import derive_positioning_signal

logger = logging.getLogger(__name__)


def run(ticker: str) -> AgentOutput:
    """Run positioning analysis (options + ownership + earnings). Cached TTL_ANALYSIS."""
    ticker = ticker.upper()
    key = f"agent:positioning:{ticker}"
    cache = agent_cache()

    cached = cache.get(key)
    if cached is not None:
        return AgentOutput.model_validate(cached)

    flow = get_options_flow(ticker)
    own = get_ownership(ticker)
    earn = get_earnings_data(ticker)
    breadth = get_breadth_snapshot()
    cboe_pc = get_cboe_pc_ratio()

    sources_active = sum(
        [
            flow.get("available", False),
            bool(own.get("institutional_holders")),
            bool(earn.get("earnings_history")),
            breadth.get("breadth_score") is not None,
            cboe_pc.get("equity_pc_ratio") is not None,
        ]
    )
    confidence = round(0.2 + (sources_active / 5) * 0.65, 2)

    positioning_label = derive_positioning_signal(
        flow_pc_volume=flow.get("pc_volume_ratio"),
        short_float_pct=own.get("short_float_pct"),
        earnings_beat_rate=earn.get("earnings_beat_rate"),
    )
    system_prompt, user_message = build_positioning_prompt(ticker, flow, own, earn, breadth, cboe_pc, positioning_label)
    # LLM required: qualitative interpretation - signal label pre-classified
    response = complete(
        system_prompt=system_prompt, user_message=user_message, tier="ANALYSIS", max_tokens=1600, temperature=0
    )

    output = AgentOutput(
        agent_name="positioning",
        content=response.content,
        sources=[
            "yfinance:options",
            "yfinance:institutional_holders",
            "yfinance:earnings",
            "yfinance:ADVN",
            "yfinance:PCCE",
        ],
        confidence=confidence,
        metadata={
            "ticker": ticker,
            "pc_volume_ratio": flow.get("pc_volume_ratio"),
            "atm_iv": flow.get("atm_iv_call"),
            "iv_skew": flow.get("iv_skew"),
            "short_float_pct": own.get("short_float_pct"),
            "next_earnings_date": earn.get("next_earnings_date"),
            "earnings_beat_rate": earn.get("earnings_beat_rate"),
            "breadth_score": breadth.get("breadth_score"),
            "mcclellan_oscillator": breadth.get("mcclellan_oscillator"),
            "cboe_equity_pc": cboe_pc.get("equity_pc_ratio"),
            "cboe_pc_sentiment": cboe_pc.get("sentiment_label"),
        },
    )
    cache.set(key, output.model_dump(mode="json"))
    return output
