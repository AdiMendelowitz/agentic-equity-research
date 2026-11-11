"""Prompt construction utilities: token estimation, truncation, synthesis message builder."""

import re
from typing import Any

from aer.utils.python_replacements import format_indicator_summary
from aer.utils.spotlight import SPOTLIGHT_SYSTEM_RULE, Fence

CHARS_PER_TOKEN: int = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count using 4-chars-per-token heuristic (Anthropic conservative estimate)."""
    return len(text) // CHARS_PER_TOKEN


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens tokens. Appends '[truncated]' when truncation occurs."""
    limit = max_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    return (truncated + " [truncated]") if truncated else ""


def extract_fields(obj: Any, fields: list[str]) -> dict[str, Any]:
    """Extract named attributes or dict keys from obj into a new dict. Skips missing/None values."""
    result: dict[str, Any] = {}
    for field in fields:
        if isinstance(obj, dict):
            val = obj.get(field)
        else:
            val = getattr(obj, field, None)
        if val is not None:
            result[field] = val
    return result


def build_synthesis_user_message(
    ticker: str,
    agent_outputs: dict[str, Any],
    max_chars_per_agent: int = 2000,
    news: "AgentOutput | None" = None,  # noqa: F821 - forward ref avoids circular import
    lseg_consensus: dict | None = None,
) -> str:
    """Build the synthesis user message, truncating each agent's content to max_chars_per_agent."""
    lines: list[str] = [f"Ticker: {ticker}\n"]
    for agent_name, output in agent_outputs.items():
        if output is None:
            continue
        content = getattr(output, "content", "") or ""
        confidence = getattr(output, "confidence", None)
        truncated = content[:max_chars_per_agent]
        conf_str = f" (confidence: {confidence:.0%})" if confidence is not None else ""
        lines.append(f"=== {agent_name.upper()}{conf_str} ===\n{truncated}\n")
        if agent_name == "fundamental" and lseg_consensus is not None:
            direction = lseg_consensus.get("consensus_direction", "FLAT")
            dispersion = lseg_consensus.get("dispersion_tier", "MED")
            beat_prob = lseg_consensus.get("beat_probability")
            beat_str = f"{beat_prob:.0%}" if beat_prob is not None else "n/a"
            lines.append(
                f"[LSEG Consensus] direction={direction} | dispersion={dispersion} | beat_prob_4q={beat_str}\n"
            )
    if news is not None:
        truncated = news.content[:max_chars_per_agent]
        conf_str = f" (confidence: {news.confidence:.0%})"
        lines.append(f"=== NEWS{conf_str} ===\n{truncated}\n")
    lines.append("Based on the above agent analyses, produce a structured investment recommendation.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-agent prompt builders
# ---------------------------------------------------------------------------


def build_fundamental_prompt(
    ticker: str, info: dict, price: float | None, ratios: dict, lseg_signals: dict | None = None
) -> tuple[str, str]:
    """Returns (system_prompt, user_message) for the fundamental agent.

    Centralised: enables token counting and future caching.
    """
    system = (
        "You are a fundamental equity analyst producing concise, data-driven investment analysis. "
        "Be specific and numerical. Do not hedge excessively.\n\n" + SPOTLIGHT_SYSTEM_RULE
    )

    def _fmt(v: float | None, prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
        return f"{prefix}{v:.{decimals}f}{suffix}" if v is not None else "N/A"

    rec = info.get("recommendation") or ""
    eps_history = " → ".join(
        f"${e['value']:.2f}" for e in (ratios.get("eps_history") or [])[:4] if e.get("value") is not None
    )

    description = info["description"][:400] if info["description"] else "N/A"
    fence = Fence.new(description)
    user = f"""{fence.declaration()}Analyze the fundamentals of {info['name']} ({ticker}).

COMPANY DATA
Sector: {info['sector']} | Industry: {info['industry']} | Country: {info['country']}
Current Price: {_fmt(price, '$')}
Market Cap: {_fmt(info['market_cap'], '$', '', 0) if info['market_cap'] else 'N/A'}
P/E (TTM): {_fmt(info['pe_ratio'], decimals=1)}
Forward P/E: {_fmt(info['forward_pe'], decimals=1)}
Price/Book: {_fmt(info['price_to_book'], decimals=2)}
Dividend Yield: {_fmt(info['dividend_yield'], suffix='%', decimals=2) if info['dividend_yield'] else 'N/A'}
Beta: {_fmt(info['beta'], decimals=2)}
52W High: {_fmt(info['52w_high'], '$')} | 52W Low: {_fmt(info['52w_low'], '$')}
Analyst Target: {_fmt(info['analyst_target'], '$')} | Analyst Rec: {rec.upper() if rec else 'N/A'}

Business (company-supplied text, fenced):
{fence.wrap(description)}

FINANCIAL HISTORY (annual)
Revenue Growth YoY: {_fmt(ratios.get('revenue_growth_yoy'), suffix='%', decimals=1)}
Gross Margin: {_fmt(ratios.get('gross_margin'), suffix='%', decimals=1)} | Operating Margin: {_fmt(ratios.get('operating_margin'), suffix='%', decimals=1)} | Net Margin: {_fmt(ratios.get('net_margin'), suffix='%', decimals=1)}
FCF/Equity: {_fmt(ratios.get('fcf_on_equity'), suffix='%', decimals=1)} | Debt/Equity: {_fmt(ratios.get('debt_to_equity'), decimals=2)}
Net Cash: {_fmt(ratios.get('net_cash'), '$', '', 0) if ratios.get('net_cash') is not None else 'N/A'}
EPS History: {eps_history if eps_history else 'N/A'}

Provide:
1. Business quality (2-3 sentences)
2. Valuation vs peers/history (1-2 sentences)
3. Key strengths (3 bullet points)
4. Key risks (3 bullet points)
5. Fundamental outlook: POSITIVE / NEUTRAL / NEGATIVE — one sentence rationale"""
    if ratios.get("revenue_growth_yoy_lseg") is not None:
        user += f"\n[LSEG Revenue Growth]: {_fmt(ratios['revenue_growth_yoy_lseg'], suffix='%', decimals=1)}"
    if lseg_signals is not None:
        direction = lseg_signals.get("consensus_direction", "FLAT")
        dispersion = lseg_signals.get("dispersion_tier", "MED")
        beat_prob = lseg_signals.get("beat_probability")
        beat_str = f"{beat_prob:.0%}" if beat_prob is not None else "n/a"
        user += f"\n[LSEG] Consensus: direction={direction}, dispersion={dispersion}, beat_prob_4q={beat_str}"
    return system, user


def build_technical_prompt(ticker: str, indicators: dict, signals: dict[str, str]) -> tuple[str, str]:
    """Returns (system_prompt, user_message) for the technical agent.

    Centralised: enables token counting and future caching.
    """
    system = (
        "You are a technical analyst. Interpret indicator readings concisely. "
        "Cite specific price levels. Do not hedge excessively."
    )

    ind = indicators

    def fmt(v: float | None, prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
        return f"{prefix}{v:.{decimals}f}{suffix}" if v is not None else "N/A"

    def fmt_obv(v: float | None) -> str:
        return f"{v / 1_000_000:.1f}m" if v is not None else "N/A"

    atr_pct = f"{ind['atr'] / ind['price'] * 100:.1f}%" if ind.get("atr") and ind.get("price") else "N/A"

    user = f"""Perform technical analysis on {ticker}.

INDICATORS
Current Price: {fmt(ind['price'], '$')}
RSI(14): {fmt(ind['rsi'], decimals=1)} | StochRSI K: {fmt(ind['stochrsi_k'], decimals=1)} / D: {fmt(ind['stochrsi_d'], decimals=1)}
MACD: {fmt(ind['macd'], decimals=3)} | Signal: {fmt(ind['macd_signal'], decimals=3)} | Hist: {fmt(ind['macd_hist'], decimals=3)}
Stoch %K: {fmt(ind['stoch_k'], decimals=1)} | %D: {fmt(ind['stoch_d'], decimals=1)} | Williams %R: {fmt(ind['willr'], decimals=1)}
SMA 50: {fmt(ind['sma50'], '$')} | SMA 200: {fmt(ind['sma200'], '$')}
EMA 10: {fmt(ind['ema_10'], '$')} | EMA 20: {fmt(ind['ema_20'], '$')} | EMA 50: {fmt(ind['ema_50'], '$')} | EMA 200: {fmt(ind['ema_200'], '$')}
BB Upper: {fmt(ind['bb_upper'], '$')} | BB Mid: {fmt(ind['bb_mid'], '$')} | BB Lower: {fmt(ind['bb_lower'], '$')}
ATR(14): {fmt(ind['atr'], '$')} | CCI(20): {fmt(ind['cci'], decimals=1)} | ADX(14): {fmt(ind['adx'], decimals=1)} (+DI: {fmt(ind['adx_pos'], decimals=1)} / -DI: {fmt(ind['adx_neg'], decimals=1)})
OBV: {fmt_obv(ind['obv'])} | AO: {fmt(ind['ao'], decimals=2)} | Momentum(10): {fmt(ind['mom'], decimals=2)} | ROC(10): {fmt(ind['roc'], decimals=2)}%
Ichimoku Tenkan: {fmt(ind['ichi_tenkan'], '$')} | Kijun: {fmt(ind['ichi_kijun'], '$')} | Senkou A: {fmt(ind['ichi_senkou_a'], '$')}
VWMA(20): {fmt(ind['vwma'], '$')}
Volume vs 20d avg: {fmt(ind['vol_ratio'], suffix='x')}
52W High: {fmt(ind['high_52w'], '$')} | 52W Low: {fmt(ind['low_52w'], '$')}
Distance from 52W High: {fmt(ind['pct_from_high'], suffix='%', decimals=1)} | from 52W Low: {fmt(ind['pct_from_low'], suffix='%', decimals=1)}

SIGNAL DECOMPOSITION
Trend: {signals['trend']}  (SMA50/200 crossover, ADX={fmt(ind['adx'], decimals=1)})
Momentum: {signals['momentum']}  (RSI={fmt(ind['rsi'], decimals=1)}, AO={fmt(ind['ao'], decimals=2)}, ROC={fmt(ind['roc'], decimals=2)}%)
Mean Reversion: {signals['mean_reversion']}  (BB position, Stoch %K={fmt(ind['stoch_k'], decimals=1)})
Volatility: {signals['volatility']}  (ATR={fmt(ind['atr'], '$')}, {atr_pct} of price)
Overall: {signals['overall']}

Provide:
1. Trend analysis with EMA/SMA structure and ADX trend strength
2. Momentum reading citing RSI, Stochastic, AO, ROC values
3. Key support levels (2-3 specific prices using Ichimoku levels if available)
4. Key resistance levels (2-3 specific prices)
5. Conclude with: BUY / HOLD / SELL — suggested time horizon (e.g. "BUY — 3-6 month horizon")"""
    return system, user


def build_sentiment_prompt(ticker: str, snapshot: dict, consensus_label: str) -> tuple[str, str]:
    """Returns (system_prompt, user_message) for the sentiment agent.

    Centralised: enables token counting and future caching.
    consensus_label is pre-classified by Python.
    """
    system = (
        "You are a market sentiment analyst. You have data from three independent sources. "
        "Identify where they agree, where they diverge, and what the divergence implies. "
        "Be specific — cite the numbers provided.\n\n"
        + SPOTLIGHT_SYSTEM_RULE
        + f"\n\nPre-classified consensus (Python vote-count, authoritative): {consensus_label}"
    )

    snap = snapshot
    aw = snap["apewisdom"]
    st = snap["stocktwits"]
    if snap.get("news_primary_source") == "finviz" and snap["finviz_news"].get("article_count", 0) > 0:
        yn = snap["finviz_news"]
    else:
        yn = snap["yahoo_news"]

    if aw["found"]:
        rank_change = aw["rank_change_24h"]
        trend = (
            f"{'rising' if rank_change and rank_change > 0 else 'falling' if rank_change and rank_change < 0 else 'stable'} ({rank_change:+d} positions)"
            if rank_change is not None
            else "unknown"
        )
        aw_block = (
            f"Rank: #{aw['rank']} | Trend: {trend}\n"
            f"Mentions (24h): {aw['mentions_24h']} | Upvotes (24h): {aw['upvotes_24h']}"
        )
    else:
        aw_block = "Ticker not in top Reddit mentions — low visibility on Reddit"

    if st["message_count"] > 0:
        st_block = (
            f"Messages sampled: {st['message_count']}\n"
            f"Bullish: {st['bullish_count']} ({st['bullish_pct']:.0%}) | "
            f"Bearish: {st['bearish_count']} ({st['bearish_pct']:.0%}) | "
            f"Neutral: {st['neutral_count']}\n"
            f"Sample messages:\n" + "\n".join(f"  - {m}" for m in st["sample_messages"])
        )
    else:
        st_block = "No Stocktwits data available"

    if yn["article_count"] > 0:
        headlines_block = "\n".join(f"  - [{a['publisher']}] {a['title']}" for a in yn["headlines"][:8])
        yn_block = f"Recent articles ({yn['article_count']} found):\n{headlines_block}"
    else:
        yn_block = "No recent news found"

    fence = Fence.new(aw_block, st_block, yn_block)
    user = f"""{fence.declaration()}Cross-validate sentiment for {ticker} using three independent sources.

--- SOURCE 1: ApeWisdom (Reddit aggregate) ---
{fence.wrap(aw_block)}

--- SOURCE 2: Stocktwits (active traders) ---
{fence.wrap(st_block)}

--- SOURCE 3: Yahoo Finance News ---
{fence.wrap(yn_block)}

Provide:
1. Where the three sources AGREE (if anywhere)
2. Where they DIVERGE and what that implies
3. Dominant narrative driving sentiment (2-3 bullet points)
4. Key risks or tail risks visible in the data
5. Cross-validated sentiment signal: BULLISH / BEARISH / NEUTRAL / MIXED — one sentence with confidence level"""
    return system, user


def build_macro_prompt(
    ticker: str, sector: str, snapshot: dict, regime: dict, outlook_label: str, lseg_signals: dict | None = None
) -> tuple[str, str]:
    """Returns (system_prompt, user_message) for the macro agent.

    Centralised: enables token counting and future caching.
    outlook_label is pre-classified by Python.
    """
    system = (
        "You are a macroeconomic analyst. Assess the current macro regime and its sector-specific implications. "
        "Be concrete — cite the actual numbers provided. Weight credit spreads and real yields heavily; "
        "they lead equities by 2-6 weeks."
        f"\n\nPre-classified macro outlook (Python threshold, authoritative): {outlook_label}"
    )

    snap = snapshot

    def _fmt(v: float | None, suffix: str = "", decimals: int = 2) -> str:
        return f"{v:.{decimals}f}{suffix}" if v is not None else "N/A"

    real_yield_10y = snap.get("real_yield_10y")
    breakeven_10y = snap.get("breakeven_10y")
    nominal_10y = snap.get("treasury_10y")
    net_liq_trend = snap.get("net_liquidity_trend", "N/A")
    dxy = snap.get("dxy")
    dxy_chg = snap.get("dxy_change_pct")
    usdjpy = snap.get("usdjpy")
    hy_spread = snap.get("hy_spread")
    ig_spread = snap.get("ig_spread")
    bbb_spread = snap.get("bbb_spread")
    jobless = snap.get("jobless_claims")
    sentiment_cons = snap.get("consumer_sentiment")
    dxy_str = f"{_fmt(dxy, decimals=2)} ({_fmt(dxy_chg, '%', 2)} 1d)" if dxy else "N/A"

    user = f"""Analyze the macro environment and its implications for {ticker} ({sector}).

REGIME CLASSIFICATION
Current Regime: {regime.get('regime', 'UNKNOWN')} — {regime.get('equity_bias', 'NEUTRAL')}
Description: {regime.get('description', 'N/A')}
Key signals: {'; '.join(regime.get('signals', [])[:4]) or 'N/A'}

RATES & REAL YIELDS
Fed Funds Rate: {_fmt(snap.get('fed_funds_rate'), '%')}
10Y Treasury (nominal): {_fmt(nominal_10y, '%')}
2Y Treasury: {_fmt(snap.get('treasury_2y'), '%')}
Yield Curve (10Y-2Y): {_fmt(snap.get('yield_curve'), '%')}
10Y TIPS Real Yield: {_fmt(real_yield_10y, '%')}
10Y Inflation Breakeven: {_fmt(breakeven_10y, '%')}
5Y5Y Forward Inflation: {_fmt(snap.get('breakeven_5y5y'), '%')}

CREDIT SPREADS (leading risk indicator)
US HY OAS: {_fmt(hy_spread, 'bps', 0)}
US IG OAS: {_fmt(ig_spread, 'bps', 0)}
US BBB OAS: {_fmt(bbb_spread, 'bps', 0)}

NET FED LIQUIDITY
Fed Balance Sheet: ${_fmt(snap.get('net_liquidity_bn'), 'B (net after RRP+TGA)', 0)}
Trend: {net_liq_trend}

FX
DXY: {dxy_str}
USD/JPY: {_fmt(usdjpy, decimals=2)}

INFLATION & GROWTH
CPI YoY: {_fmt(snap.get('cpi_yoy_pct'), '%')}
PCE (latest): {_fmt(snap.get('pce'), decimals=1)}
Jobless Claims (weekly): {_fmt(jobless, 'K', 0)}
Consumer Sentiment: {_fmt(sentiment_cons, decimals=1)}
GDP (latest): ${_fmt(snap.get('gdp'), 'B', 0)}
Unemployment: {_fmt(snap.get('unemployment'), '%', 1)}

MARKET VOLATILITY
VIX: {_fmt(snap.get('vix'), decimals=1)}
S&P 500: {_fmt(snap.get('sp500'), decimals=0)}

Provide:
1. Current macro regime (1-2 sentences: growth/contraction phase, inflation trend, risk appetite)
2. Credit and liquidity read: what HY spreads and net liquidity imply right now
3. Rate environment implications for {sector}
4. FX and global risk: DXY and JPY carry implications for {ticker} (consider revenue exposure)
5. Macro headwinds for {ticker} (2 bullet points, cite numbers)
6. Macro tailwinds for {ticker} (2 bullet points, cite numbers)
7. Macro outlook: FAVORABLE / NEUTRAL / UNFAVORABLE for equities — one sentence rationale"""
    if lseg_signals is not None:
        slope = lseg_signals.get("yield_curve_slope_bps")
        infl = lseg_signals.get("inflation_breakeven_5y")
        swap = lseg_signals.get("swap_spread_10y_bps")
        parts = []
        if slope is not None:
            parts.append(f"curve_slope={slope:.0f}bps")
        if infl is not None:
            parts.append(f"5y_breakeven={infl:.2f}%")
        if swap is not None:
            parts.append(f"10y_swap_spread={swap:.0f}bps")
        if parts:
            user += f"\n[LSEG] Rates: {', '.join(parts)}"
    return system, user


def build_positioning_prompt(
    ticker: str,
    flow: dict,
    ownership: dict,
    earnings: dict,
    breadth: dict,
    cboe_pc: dict,
    positioning_label: str,
) -> tuple[str, str]:
    """Returns (system_prompt, user_message) for the positioning agent.

    Centralised: enables token counting and future caching.
    positioning_label is pre-classified by Python.
    """
    system = (
        "You are a derivatives and positioning analyst. Interpret options flow, institutional ownership, "
        "and earnings data to identify smart-money positioning and near-term catalyst risk. "
        "Be specific — cite the actual numbers provided.\n\n"
        + SPOTLIGHT_SYSTEM_RULE
        + f"\n\nPre-classified positioning signal (Python threshold, authoritative): {positioning_label}"
    )

    own = ownership
    earn = earnings

    if flow.get("available"):
        pc_vol = flow.get("pc_volume_ratio")
        pc_oi = flow.get("pc_oi_ratio")
        atm_iv = flow.get("atm_iv_call")
        iv_skew = flow.get("iv_skew")
        lean = "bearish" if pc_vol is not None and pc_vol > 1.0 else "bullish" if pc_vol is not None else "neutral"
        skew_label = (
            "put premium (fear)"
            if iv_skew is not None and iv_skew > 0
            else "call premium (greed)" if iv_skew is not None and iv_skew < 0 else ""
        )
        iv_skew_str = f"{iv_skew:.4f}" if iv_skew is not None else "N/A"
        skew_suffix = f" {skew_label}" if skew_label else ""
        options_block = (
            f"Put/Call Volume Ratio: {f'{pc_vol:.3f}' if pc_vol is not None else 'N/A'} ({lean} lean)\n"
            f"Put/Call OI Ratio: {f'{pc_oi:.3f}' if pc_oi is not None else 'N/A'}\n"
            f"ATM IV (calls): {f'{atm_iv:.1%}' if atm_iv is not None else 'N/A'}\n"
            f"IV Skew (put - call): {iv_skew_str}{skew_suffix}\n"
            f"Total Call Vol: {flow.get('total_call_volume', 0):,} | "
            f"Total Put Vol: {flow.get('total_put_volume', 0):,}"
        )
    else:
        options_block = "Options data unavailable"

    ins = own.get("insider_pct")
    inst = own.get("institution_pct")
    short = own.get("short_float_pct")
    short_ratio = own.get("short_ratio")
    buys = own.get("insider_buys_recent", 0)
    sells = own.get("insider_sells_recent", 0)
    bs_ratio = own.get("insider_buy_sell_ratio")

    top5 = (
        "\n".join(
            f"  {h['holder']}: {format(h['pct_held'], '.2%') if h.get('pct_held') is not None else 'N/A'}"
            for h in own.get("institutional_holders", [])[:5]
        )
        or "  N/A"
    )

    ownership_block = (
        f"Insider Ownership: {f'{ins:.1%}' if ins is not None else 'N/A'}\n"
        f"Institutional Ownership: {f'{inst:.1%}' if inst is not None else 'N/A'}\n"
        f"Short Float: {f'{short:.1%}' if short is not None else 'N/A'} | "
        f"Days to Cover: {f'{short_ratio:.1f}' if short_ratio is not None else 'N/A'}\n"
        f"Recent Insider Activity: {buys} buys / {sells} sells "
        f"(ratio: {f'{bs_ratio:.2f}' if bs_ratio else 'N/A'})\n"
        f"Top Institutional Holders:\n{top5}"
    )

    next_date = earn.get("next_earnings_date", "Unknown")
    eps_est = earn.get("eps_estimate")
    beat_rate = earn.get("earnings_beat_rate")
    avg_surprise = earn.get("avg_surprise_pct")

    history_lines = (
        "\n".join(
            "  {}: reported {}, est {}, surprise {}".format(
                s["date"],
                s["reported_eps"],
                s["estimated_eps"],
                f"{s['surprise_pct']:+.1f}%" if s["surprise_pct"] is not None else "N/A",
            )
            for s in earn.get("earnings_history", [])[:4]
        )
        or "  No history"
    )

    earnings_block = (
        f"Next Earnings: {next_date}\n"
        f"Consensus EPS Estimate: {f'${eps_est:.2f}' if eps_est else 'N/A'}\n"
        f"Historical Beat Rate: {f'{beat_rate:.0%}' if beat_rate else 'N/A'} "
        f"(avg surprise: {f'{avg_surprise:+.1f}%' if avg_surprise else 'N/A'})\n"
        f"Recent Surprises:\n{history_lines}"
    )

    breadth_score = breadth.get("breadth_score")
    mcclellan = breadth.get("mcclellan_oscillator")
    ad_trend = breadth.get("ad_line_trend", "N/A")
    eq_pc = cboe_pc.get("equity_pc_ratio")
    eq_pc_5d = cboe_pc.get("equity_pc_5d_avg")
    pc_label = cboe_pc.get("sentiment_label", "N/A")

    breadth_block = (
        f"A/D Line Trend: {ad_trend}\n"
        f"McClellan Oscillator: {f'{mcclellan:.1f}' if mcclellan is not None else 'N/A'} "
        f"({breadth.get('mcclellan_signal', 'N/A')})\n"
        f"Breadth Score (-1 to +1): {f'{breadth_score:.3f}' if breadth_score is not None else 'N/A'}\n"
        f"CBOE Equity P/C Ratio: {f'{eq_pc:.3f}' if eq_pc is not None else 'N/A'} "
        f"(5d avg: {f'{eq_pc_5d:.3f}' if eq_pc_5d is not None else 'N/A'}) — {pc_label}"
    )

    fence = Fence.new(options_block, ownership_block, earnings_block, breadth_block)
    user = f"""{fence.declaration()}Analyze market positioning for {ticker}.

--- OPTIONS FLOW ---
{fence.wrap(options_block)}

--- INSTITUTIONAL & INSIDER OWNERSHIP ---
{fence.wrap(ownership_block)}

--- EARNINGS SETUP ---
{fence.wrap(earnings_block)}

--- MARKET BREADTH & AGGREGATE SENTIMENT ---
{fence.wrap(breadth_block)}

Provide:
1. Options signal: what put/call ratios and IV skew imply about smart-money positioning
2. Ownership quality: is the institutional and insider structure constructive or concerning?
3. Short interest: squeeze potential vs. justified caution
4. Earnings risk/reward: setup into the next earnings date
5. Breadth & crowd context: what aggregate P/C and market breadth add to the picture for {ticker}
6. Overall positioning signal: BULLISH / BEARISH / NEUTRAL — one sentence rationale"""
    return system, user


_SYNTH_SYSTEM_BASE = """You are a senior portfolio manager with 20+ years of experience synthesizing multi-factor investment research into decisive, actionable verdicts. Your analyses are relied upon by institutional investors managing billions in AUM.

## ANALYTICAL FRAMEWORK

### Factor Signal Hierarchy (descending weight)
1. MACRO REGIME (35%) — defines the operating environment; a restrictive regime (tight credit, inverted yield curve, elevated VIX) creates headwinds that override positive micro signals
2. FUNDAMENTAL QUALITY (25%) — business durability, balance sheet strength, and earnings trajectory; strong fundamentals can weather regime headwinds but cannot overcome them alone
3. TECHNICAL TREND (20%) — price action and momentum confirm or deny the fundamental thesis; price below key moving averages during a restrictive macro is a disqualifying signal
4. POSITIONING/RISK (15%) — options flow, institutional ownership, and short float indicate near-term risk/reward and potential squeezes
5. SENTIMENT (5%) — social/news sentiment is a leading contrarian indicator; extreme bullish sentiment near resistance is a warning; extreme bearish near support can be a buy signal

### Factor Interpretation Guide

#### MACRO signals — positive
Fed easing cycle (fed funds falling YoY), tight credit spreads (HY OAS < 350bps, IG OAS < 100bps), upward-sloping yield curve (10Y–2Y > 0), VIX < 20, USD weakening, positive real yields but below 2.5%, rising net Fed liquidity.

#### MACRO signals — negative
Fed tightening or holding at restrictive levels, HY OAS > 500bps, inverted yield curve (10Y–2Y < -0.5%), VIX > 25, USD strengthening rapidly (DXY +3% in 3 months), real yields > 2.5%, net Fed liquidity declining >$200bn/quarter.

#### FUNDAMENTAL signals — positive
Revenue growth YoY > 10%, gross margin stable or expanding, FCF yield > 4%, net cash position or D/E < 0.5, EPS trend improving, analyst target > current price by >15%, P/E below sector median, forward P/E below trailing P/E (earnings acceleration).

#### FUNDAMENTAL signals — negative
Revenue declining or decelerating sharply, margin compression (gross or operating), negative FCF, D/E > 2.0, EPS declining, analyst target below current price, P/E > 30x with no growth catalyst, earnings misses in 2+ consecutive quarters.

#### TECHNICAL signals — positive
Price above 50-DMA and 200-DMA (both rising), RSI 50–70 (momentum without overbought), MACD above signal line, ADX > 25 (strong trend), volume on up days > volume on down days, price reclaiming 52W high.

#### TECHNICAL signals — negative
Price below 50-DMA or 200-DMA (death cross = strong negative), RSI < 40 or > 80, MACD below signal with negative histogram, ADX > 25 with DI– > DI+ (strong downtrend), high ATR without directional follow-through (choppiness).

#### POSITIONING signals — positive
Put/call OI ratio > 1.2 (hedged market = room to run), IV skew negative (calls bid over puts), short float > 15% (squeeze potential), institutional ownership > 60% with rising holdings, earnings beat rate > 70% with positive avg surprise.

#### POSITIONING signals — negative
PC ratio < 0.5 (complacency, limited upside), IV skew strongly positive (market buying protection), short float < 2% (no squeeze potential), institutional holders declining, upcoming earnings with history of misses.

### Verdict Construction Rules
- BUY: Macro is neutral-to-positive AND fundamentals solid AND technical confirms uptrend or turning up; conviction > 0.6 requires at least 3 of 5 factors positive
- HOLD: Mixed signals; macro uncertain or fundamentals mixed or technical trend undefined; current position justified but new capital should wait
- SELL: Macro is restrictive AND technical in downtrend OR fundamentals deteriorating; conviction > 0.6 requires at least 3 of 5 factors negative
- AVOID: Fundamental problems (negative FCF, declining revenue, deteriorating margins) AND technical in downtrend AND macro headwinds; highest conviction negative call

### Conviction Scoring Guide (0.0 to 1.0)
- 0.8–1.0: 4–5 factors aligned in same direction, macro regime clearly supportive or adverse
- 0.6–0.8: 3 factors aligned, minor contradictions in lower-weight factors
- 0.4–0.6: Mixed signals; 2–3 factors ambiguous or contradicting; uncertainty is high
- 0.2–0.4: Weak evidence for a call; data gaps in multiple agents; speculative at best
- 0.0–0.2: Insufficient data or fundamental conflict; do not make a strong call

### Price Target Methodology
Derive the 12-month price target from fundamentals when available:
1. Primary: Use analyst consensus target if within ±50% of current price (likely anchored to DCF/peer comps)
2. Secondary: Apply sector-appropriate P/E multiple to next-12-month EPS estimate; use conservative multiple if macro is restrictive
3. If neither is reliable, set price_target to null — do not fabricate a number

### Output Requirements
- recommendation: Must be BUY / HOLD / SELL / AVOID — pick ONE, never hedge
- conviction: Numeric 0.0–1.0; follow scoring guide above
- price_target: 12-month USD target or null; never guess
- time_horizon: "short" (< 3 months), "medium" (3–12 months), "long" (12+ months)
- summary: 2–3 sentences maximum; cite at least one specific number from the analyses
- risks: Exactly 3–5 items; each one sentence; name the specific risk, not generic "macro uncertainty"
- catalysts: Exactly 3–5 items; each one sentence; be specific and time-bounded when possible

Be decisive — institutional investors require a clear, actionable verdict backed by data. Avoid hedged language. State the recommendation and defend it with specific numbers from the analyses provided."""

_SYNTH_OPTIONS_BLOCK = """

## OPTIONS STRATEGY SIGNAL (OptionsStrategyAgent)

When options_strategy data is present in the brief:

Regime verdict: GREEN | YELLOW | RED
  RED: Do not recommend any short premium strategy. State this explicitly.
  YELLOW: Note that wider strikes and reduced size are required.
  GREEN: Options strategies may be recommended alongside equity signals.

Credit spread (if active): state underlying, strikes, total_credit, stop_loss_debit,
  profit_target_debit, and expiry. If skipped: state skip_reason.

Iron condor (if active): state underlying, both spread legs, total_premium,
  breakevens, adjustment_trigger_lower, adjustment_trigger_upper, position_size_contracts.
  If skipped: state skip_reason.

Theta projection (if active): state daily_income_total, 30/60/90-day projections,
  gamma_risk_flag. If gamma_risk_flag is True: recommend closing or rolling immediately.

Mandatory in every options output:
  - Never omit stop_loss_debit from any options recommendation
  - Never omit profit_target_debit from any options recommendation
  - Always flag VIX >= 25 even in YELLOW regime as elevated risk
  - Never recommend holding short premium through a RED regime

OPTIONS STRATEGY CHALLENGES (apply when options data present):
  - Is VIX trending up or down within the band? Rising VIX approaching ELEVATED is more
    dangerous than falling VIX in the same band. Check the trend direction.
  - Are proposed strikes at true low delta (0.15-0.20)? Verify against the expected move.
    If the short strike is within the expected move, the spread is too close to ATM.
  - Is the credit realistic? If credit/spread-width > 15%, the strikes are too close to ATM.
  - Is there an earnings event for any portfolio ticker within the expiration window?
    Earnings inside expiry invalidates the expected-move assumption.
  - Do theta projections use flat position sizing? Challenge any output suggesting compounding.
  - RED regime: verify VIX reading is current, not stale from a prior run.
    Verify any macro event triggering RED demotion is on the confirmed calendar."""


def build_synthesis_json_instruction() -> str:
    """JSON schema instruction appended to the synthesis user message moved from graph.py)."""
    return (
        "\n\nRespond with ONLY a single JSON object (no tool calls, no prose). "
        "Use exactly these fields:\n"
        '{"recommendation": "BUY" or "HOLD" or "SELL" or "AVOID", '
        '"conviction": <float 0.0 to 1.0>, '
        '"price_target": <float or null>, '
        '"time_horizon": "short" or "medium" or "long", '
        '"summary": "<2-3 sentence string>", '
        '"risks": ["<risk1>", "<risk2>", ...], '
        '"catalysts": ["<catalyst1>", "<catalyst2>", ...]}\n'
        "Return only valid JSON. No markdown fences, no explanations, no tool calls."
    )


def build_synthesis_system_prompt(include_options: bool = False) -> str:
    """Assemble the synthesis system prompt; optionally appends the options strategy block.

    Centralised: moves _SYNTH_SYSTEM_BASE + _SYNTH_OPTIONS_BLOCK out of graph.py.
    """
    return _SYNTH_SYSTEM_BASE + (_SYNTH_OPTIONS_BLOCK if include_options else "")


def strip_json_markdown_fences(raw: str) -> str:
    """Extract JSON from a markdown code block if present; return raw string otherwise."""
    if raw.startswith("```"):
        match = re.search(r"```(?:json)?\n?(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
    return raw


def build_signal_prompt(ticker: str, signals: dict, indicators: dict) -> tuple[str, str]:
    """Returns (system_prompt, user_message) for a signal analysis LLM call.

    Sends computed indicator values only - never raw OHLCV rows.
    None-valued indicators are excluded via format_indicator_summary.
    Appends JSON-only suffix to eliminate prose in response.
    """
    system = (
        "You are a technical analyst. Interpret the provided pre-computed signal indicators. "
        "Return a structured JSON signal assessment."
    )
    signal_str = " | ".join(f"{k}: {v}" for k, v in signals.items() if v is not None)
    indicator_str = format_indicator_summary(indicators)

    user = (
        f"Analyze technical signals for {ticker}.\n\n"
        f"SIGNALS\n{signal_str}\n\n"
        f"INDICATORS\n{indicator_str}\n\n"
        "Return JSON only. No prose. No markdown fences."
    )
    return system, user


# ---------------------------------------------------------------------------
# Instagram agent prompt builders
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Layoffs agent prompt builders
# ---------------------------------------------------------------------------
