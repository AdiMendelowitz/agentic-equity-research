"""Fundamental analysis agent."""

import logging

from aer.data.financials import get_key_ratios
from aer.data.prices import get_company_info, get_current_price
from aer.models import AgentOutput
from aer.utils.cache import agent_cache
from aer.utils.llm_client import complete
from aer.utils.prompt_builders import build_fundamental_prompt

logger = logging.getLogger(__name__)


def run(ticker: str) -> AgentOutput:
    """Run fundamental analysis for ticker. Cached for TTL_ANALYSIS seconds."""
    ticker = ticker.upper()
    key = f"agent:fundamental:{ticker}"
    cache = agent_cache()

    cached = cache.get(key)
    if cached is not None:
        return AgentOutput.model_validate(cached)

    info = get_company_info(ticker)
    price = get_current_price(ticker)
    ratios = get_key_ratios(ticker)

    system_prompt, user_message = build_fundamental_prompt(ticker, info, price, ratios, lseg_signals={})
    # LLM required: qualitative narrative - business quality, valuation, risks, catalysts
    response = complete(
        system_prompt=system_prompt, user_message=user_message, tier="ANALYSIS", max_tokens=1400, temperature=0
    )

    data_fields_filled = sum(1 for k, v in info.items() if k != "ticker" and v not in (None, "", 0))
    confidence = round(min(0.9, data_fields_filled / max(len(info) - 1, 1)), 2)

    sources = [f"yfinance:{ticker}"]

    output = AgentOutput(
        agent_name="fundamental",
        content=response.content,
        sources=sources,
        confidence=confidence,
        metadata={"ticker": ticker, "price": price, "sector": info.get("sector")},
    )
    cache.set(key, output.model_dump(mode="json"))
    return output
