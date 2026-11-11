"""Technical analysis agent using pandas-ta indicators."""

import logging
from typing import Any

from aer.data.prices import get_price_history_raw, get_raw_history_last_year
from aer.indicators.technical import compute_indicators
from aer.models import AgentOutput
from aer.utils.cache import agent_cache
from aer.utils.constants import (
    TECH_ADX_TREND_MIN,
    TECH_ATR_HIGH_VOL_PCT,
    TECH_ATR_LOW_VOL_PCT,
    TECH_BB_WIDTH_HIGH_PCT,
    TECH_MOMENTUM_MAJORITY_FRAC,
    TECH_MOMENTUM_MIDLINE,
    TECH_MOMENTUM_MIN_AGREEING,
    TECH_OVERALL_CONSENSUS_MIN,
    TECH_STOCH_OVERBOUGHT,
    TECH_STOCH_OVERSOLD,
)
from aer.utils.llm_client import complete
from aer.utils.prompt_builders import build_technical_prompt

logger = logging.getLogger(__name__)


def _classify_signals(ind: dict[str, Any]) -> dict[str, str]:
    """Deterministic rule-based sub-signal classification. No LLM."""

    def trend_signal() -> str:
        sma50, sma200, adx = ind.get("sma50"), ind.get("sma200"), ind.get("adx")
        if sma50 is None or sma200 is None:
            return "NEUTRAL"
        if sma50 > sma200 and (adx is None or adx >= TECH_ADX_TREND_MIN):
            return "BULLISH"
        if sma50 < sma200 and (adx is None or adx >= TECH_ADX_TREND_MIN):
            return "BEARISH"
        return "NEUTRAL"

    def momentum_signal() -> str:
        rsi, stoch_k, roc, ao = ind.get("rsi"), ind.get("stoch_k"), ind.get("roc"), ind.get("ao")
        present = [v for v in [rsi, stoch_k, roc, ao] if v is not None]
        if not present:
            return "NEUTRAL"
        # RSI and stoch_k: threshold at 50. ROC and AO: strictly positive is bullish,
        # strictly negative is bearish, zero is neutral (contributes to neither count).
        bullish = sum(
            [
                rsi is not None and rsi > TECH_MOMENTUM_MIDLINE,
                stoch_k is not None and stoch_k > TECH_MOMENTUM_MIDLINE,
                roc is not None and roc > 0.0,
                ao is not None and ao > 0.0,
            ]
        )
        bearish = sum(
            [
                rsi is not None and rsi <= TECH_MOMENTUM_MIDLINE,
                stoch_k is not None and stoch_k <= TECH_MOMENTUM_MIDLINE,
                roc is not None and roc < 0.0,
                ao is not None and ao < 0.0,
            ]
        )
        n = len(present)
        if bullish >= max(TECH_MOMENTUM_MIN_AGREEING, n):
            return "BULLISH"
        if bearish >= max(TECH_MOMENTUM_MIN_AGREEING, n):
            return "BEARISH"
        # majority among present indicators
        return (
            "BULLISH"
            if bullish / n >= TECH_MOMENTUM_MAJORITY_FRAC
            else ("BEARISH" if bearish / n >= TECH_MOMENTUM_MAJORITY_FRAC else "NEUTRAL")
        )

    def mean_reversion_signal() -> str:
        price = ind.get("price")
        bb_upper, bb_lower = ind.get("bb_upper"), ind.get("bb_lower")
        stoch_k = ind.get("stoch_k")
        if price is not None and bb_lower is not None and price < bb_lower:
            return "BULLISH"
        if price is not None and bb_upper is not None and price > bb_upper:
            return "BEARISH"
        if stoch_k is not None and stoch_k < TECH_STOCH_OVERSOLD:
            return "BULLISH"
        if stoch_k is not None and stoch_k > TECH_STOCH_OVERBOUGHT:
            return "BEARISH"
        return "NEUTRAL"

    def volatility_signal() -> str:
        atr, price = ind.get("atr"), ind.get("price")
        bb_upper, bb_lower = ind.get("bb_upper"), ind.get("bb_lower")
        if atr is not None and price:  # price truthy guards divide-by-zero deliberately
            atr_pct = (atr / price) * 100
            if atr_pct > TECH_ATR_HIGH_VOL_PCT:
                return "HIGH"
            if atr_pct < TECH_ATR_LOW_VOL_PCT:
                return "LOW"
        if bb_upper is not None and bb_lower is not None and price:
            if (bb_upper - bb_lower) / price * 100 > TECH_BB_WIDTH_HIGH_PCT:
                return "HIGH"
        return "NORMAL"

    trend = trend_signal()
    momentum = momentum_signal()
    mean_rev = mean_reversion_signal()
    bullish = sum(s == "BULLISH" for s in [trend, momentum, mean_rev])
    bearish = sum(s == "BEARISH" for s in [trend, momentum, mean_rev])
    overall = (
        "BULLISH"
        if bullish >= TECH_OVERALL_CONSENSUS_MIN
        else ("BEARISH" if bearish >= TECH_OVERALL_CONSENSUS_MIN else "NEUTRAL")
    )
    return {
        "trend": trend,
        "momentum": momentum,
        "mean_reversion": mean_rev,
        "volatility": volatility_signal(),
        "overall": overall,
    }


def run(ticker: str, period: str = "1y") -> AgentOutput:
    """Run technical analysis for ticker. Cached for TTL_ANALYSIS seconds."""
    ticker = ticker.upper()
    key = f"agent:technical:{ticker}:{period}"
    cache = agent_cache()

    cached = cache.get(key)
    if cached is not None:
        return AgentOutput.model_validate(cached)

    # indicators on the dividend-unadjusted path. The default 1y reads the prefetch-warmed
    # 3y raw key and slices it; other periods fetch that period directly.
    df = (
        get_raw_history_last_year(ticker)
        if period == "1y"
        else get_price_history_raw(ticker, period=period, interval="1d")
    )

    if df.empty:
        return AgentOutput(
            agent_name="technical",
            content="Insufficient price data for technical analysis.",
            confidence=0.0,
            metadata={"ticker": ticker, "error": "no_data"},
        )

    ind = compute_indicators(df.copy())
    signals = _classify_signals(ind)
    indicators_available = sum(1 for v in ind.values() if v is not None)
    confidence = round(min(0.9, indicators_available / len(ind)), 2)

    system_prompt, user_message = build_technical_prompt(ticker, ind, signals)
    # LLM required: qualitative narrative - trend, momentum, support/resistance levels
    response = complete(
        system_prompt=system_prompt, user_message=user_message, tier="ANALYSIS", max_tokens=1200, temperature=0
    )

    output = AgentOutput(
        agent_name="technical",
        content=response.content,
        sources=[f"yfinance:{ticker}:{period}"],
        confidence=confidence,
        metadata={"ticker": ticker, "indicators": ind, "signals": signals},
    )
    cache.set(key, output.model_dump(mode="json"))
    return output
