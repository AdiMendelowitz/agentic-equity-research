"""Macro context agent - FRED rates, credit spreads, real yields, FX, net liquidity, regime."""

import logging

from aer.data.macro import get_macro_snapshot
from aer.data.prices import get_company_metadata
from aer.data.regime import classify_regime
from aer.models import AgentOutput
from aer.utils.cache import agent_cache
from aer.utils.llm_client import complete
from aer.utils.prompt_builders import build_macro_prompt
from aer.utils.python_replacements import derive_macro_outlook

logger = logging.getLogger(__name__)


def run(ticker: str) -> AgentOutput:
    """Run macro analysis for ticker. Cached for TTL_ANALYSIS seconds."""
    ticker = ticker.upper()
    key = f"agent:macro:{ticker}"
    cache = agent_cache()

    cached = cache.get(key)
    if cached is not None:
        return AgentOutput.model_validate(cached)

    meta = get_company_metadata(ticker)
    sector = meta.get("sector", "Unknown")
    snap = get_macro_snapshot()
    regime = classify_regime(snap)

    available = sum(1 for v in snap.values() if v is not None)
    confidence = round(min(0.9, 0.5 + (available / max(len(snap), 1)) * 0.4), 2)

    outlook_label = derive_macro_outlook(regime)
    system_prompt, user_message = build_macro_prompt(ticker, sector, snap, regime, outlook_label, lseg_signals={})
    # LLM required: qualitative narrative and risk factors - outlook label pre-classified
    response = complete(
        system_prompt=system_prompt, user_message=user_message, tier="ANALYSIS", max_tokens=2400, temperature=0
    )

    sources = [
        "fredapi:FEDFUNDS",
        "fredapi:CPIAUCSL",
        "fredapi:DGS10",
        "fredapi:VIXCLS",
        "fredapi:BAMLH0A0HYM2OAS",
        "fredapi:BAMLC0A0CM",
        "fredapi:DFII10",
        "fredapi:T10YIE",
        "fredapi:WALCL",
        "fredapi:RRPONTSYD",
        "fredapi:WTREGEN",
        "yfinance:DX-Y.NYB",
        "yfinance:JPY=X",
    ]
    output = AgentOutput(
        agent_name="macro",
        content=response.content,
        sources=sources,
        confidence=confidence,
        metadata={
            "ticker": ticker,
            "sector": sector,
            "macro_snapshot": snap,
            "regime": regime,
        },
    )
    cache.set(key, output.model_dump(mode="json"))
    return output
