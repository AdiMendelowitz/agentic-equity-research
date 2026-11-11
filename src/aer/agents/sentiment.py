"""Sentiment analysis agent - cross-validates ApeWisdom, Stocktwits, and Yahoo Finance news."""

import logging

from aer.data.sentiment_feeds import get_sentiment_snapshot
from aer.models import AgentOutput
from aer.utils.cache import agent_cache
from aer.utils.llm_client import complete
from aer.utils.prompt_builders import build_sentiment_prompt
from aer.utils.python_replacements import derive_sentiment_consensus

logger = logging.getLogger(__name__)


def _score_confidence(snap: dict) -> float:
    news_active = (
        snap.get("finviz_news", {}).get("article_count", 0) > 0
        or snap.get("yahoo_news", {}).get("article_count", 0) > 0
    )
    sources_active = sum(
        [
            snap.get("apewisdom", {}).get("found", False),
            snap.get("stocktwits", {}).get("message_count", 0) > 0,
            news_active,
        ]
    )
    return round(0.2 + (sources_active / 3) * 0.65, 2)


def run(ticker: str) -> AgentOutput:
    """Run cross-validated sentiment analysis. Cached for TTL_ANALYSIS seconds."""
    ticker = ticker.upper()
    key = f"agent:sentiment:{ticker}"
    cache = agent_cache()

    cached = cache.get(key)
    if cached is not None:
        return AgentOutput.model_validate(cached)
    snap = get_sentiment_snapshot(ticker)
    confidence = _score_confidence(snap)

    if confidence <= 0.2:
        return AgentOutput(
            agent_name="sentiment",
            content="No sentiment data available from any source (Reddit, Stocktwits, news).",
            confidence=0.0,
            metadata={"ticker": ticker, "sources_active": 0},
        )

    # Pre-classify consensus verdict from numeric source data - LLM handles narrative only
    st = snap.get("stocktwits", {})
    aw = snap.get("apewisdom", {})
    _sources: dict[str, str] = {}
    if st.get("message_count", 0) > 0:
        bullish_pct: float = st.get("bullish_pct", 0.0) or 0.0
        _sources["stocktwits"] = "BULLISH" if bullish_pct > 0.6 else ("BEARISH" if bullish_pct < 0.3 else "NEUTRAL")
    if aw.get("found"):
        rank_change = aw.get("rank_change_24h")
        _sources["apewisdom"] = (
            "BULLISH"
            if rank_change and rank_change > 0
            else ("BEARISH" if rank_change and rank_change < 0 else "NEUTRAL")
        )
    news_count = max(
        snap.get("finviz_news", {}).get("article_count", 0),
        snap.get("yahoo_news", {}).get("article_count", 0),
    )
    if news_count > 0:
        _sources["news"] = "NEUTRAL"  # conservative default; narrative assessed by LLM
    consensus_label = derive_sentiment_consensus(_sources)

    system_prompt, user_message = build_sentiment_prompt(ticker, snap, consensus_label)
    # LLM required: qualitative narrative only - verdict pre-classified
    response = complete(
        system_prompt=system_prompt, user_message=user_message, tier="ANALYSIS", max_tokens=1400, temperature=0
    )

    output = AgentOutput(
        agent_name="sentiment",
        content=response.content,
        sources=["apewisdom.io", "stocktwits.com", "finance.yahoo.com"],
        confidence=confidence,
        metadata={
            "ticker": ticker,
            "apewisdom_rank": snap.get("apewisdom", {}).get("rank"),
            "stocktwits_bullish_pct": snap.get("stocktwits", {}).get("bullish_pct"),
            "news_count": snap.get("yahoo_news", {}).get("article_count", 0),
        },
    )
    cache.set(key, output.model_dump(mode="json"))
    return output
