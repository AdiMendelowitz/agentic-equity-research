"""Regression guard: canonical Python threshold functions that must never be replaced with LLM calls."""

from collections import Counter

import pandas as pd

STRONG_BUY_THRESHOLD: float = 6.0
BUY_THRESHOLD: float = 3.0
WATCH_THRESHOLD: float = 1.5

VIX_LOW: float = 15.0
VIX_NORMAL: float = 20.0
VIX_ELEVATED: float = 30.0

TREND_THRESHOLD_PCT: float = 0.01  # 1% of absolute mean


def signal_label(score: float) -> str:
    """Map insider score to signal label. Canonical implementation."""
    if score >= STRONG_BUY_THRESHOLD:
        return "STRONG_BUY"
    if score >= BUY_THRESHOLD:
        return "BUY"
    if score >= WATCH_THRESHOLD:
        return "WATCH"
    return "NOISE"


def classify_vix_regime(vix: float) -> str:
    """Classify VIX into regime label. Pure Python - never use LLM for this."""
    if vix < VIX_LOW:
        return "LOW_FEAR"
    if vix < VIX_NORMAL:
        return "NORMAL"
    if vix < VIX_ELEVATED:
        return "ELEVATED"
    return "HIGH_FEAR"


def pc_ratio_lean(ratio: float) -> str:
    """Classify put/call ratio into market lean. Pure Python - never use LLM for this."""
    return "bearish" if ratio > 1.0 else "bullish"


def signal_action_from_classification(overall: str) -> str:
    """Map technical classifier overall label to portfolio action string.

    # LLM required: N/A - deterministic mapping replaces EXTRACTION-tier LLM call
    BULLISH -> BUY, BEARISH -> SELL, NEUTRAL or unknown -> HOLD.
    """
    _MAP = {"BULLISH": "BUY", "BEARISH": "SELL"}
    return _MAP.get(str(overall).upper(), "HOLD")


def derive_macro_outlook(regime: dict) -> str:
    """Map regime dict to FAVORABLE / NEUTRAL / UNFAVORABLE equity outlook.

    # LLM required: N/A - replaces LLM categorical verdict in macro agent
    Reads regime['equity_bias']: BULLISH -> FAVORABLE; BEARISH* -> UNFAVORABLE; else NEUTRAL.
    """
    bias = str(regime.get("equity_bias", "")).upper()
    if "BEARISH" in bias:
        return "UNFAVORABLE"
    if "BULLISH" in bias:
        return "FAVORABLE"
    return "NEUTRAL"


def derive_sentiment_consensus(sources: dict) -> str:
    """Map {source_name: 'BULLISH'|'BEARISH'|'NEUTRAL'} to consensus label.

    # LLM required: N/A - deterministic vote-counting replaces LLM verdict
    BULLISH/BEARISH only if >=2 of the provided sources agree; otherwise MIXED.
    Returns one of: BULLISH | BEARISH | NEUTRAL | MIXED.
    """
    counts = Counter(str(v).upper() for v in sources.values())
    for label in ("BULLISH", "BEARISH", "NEUTRAL"):
        if counts.get(label, 0) >= 2:
            return label
    return "MIXED"


def derive_positioning_signal(
    flow_pc_volume: float | None,
    short_float_pct: float | None,
    earnings_beat_rate: float | None,
) -> str:
    """Map positioning numerics to BULLISH / BEARISH / NEUTRAL.

    # LLM required: N/A - threshold logic replaces LLM verdict in positioning agent
    Heuristic: pc_vol > 1.2 AND earnings_beat > 0.7 -> BULLISH (squeeze setup);
               pc_vol < 0.5 AND earnings_beat < 0.4 -> BEARISH; else NEUTRAL.
    short_float_pct informational only (not yet in heuristic - needs calibration data).
    """
    pc = flow_pc_volume if flow_pc_volume is not None else 1.0
    beat = earnings_beat_rate if earnings_beat_rate is not None else 0.5
    if pc > 1.2 and beat > 0.7:
        return "BULLISH"
    if pc < 0.5 and beat < 0.4:
        return "BEARISH"
    return "NEUTRAL"


def compute_trend(series: pd.Series, n: int = 3) -> str:
    """Classify the recent direction of a numeric series.

    # LLM required: N/A - delta arithmetic on a pandas Series is deterministic Python.
    Returns RISING / FALLING / FLAT / INSUFFICIENT_DATA.
    Threshold: 1% of the absolute mean of the tail window.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return "INSUFFICIENT_DATA"
    recent = clean.tail(n)
    delta = recent.iloc[-1] - recent.iloc[0]
    threshold = abs(recent.mean()) * TREND_THRESHOLD_PCT
    if delta > threshold:
        return "RISING"
    if delta < -threshold:
        return "FALLING"
    return "FLAT"


def format_indicator_summary(indicators: dict) -> str:
    """Format an indicator dict as 'KEY: value | KEY: value ...', skipping None values.

    # LLM required: N/A - string formatting from a known-structure dict is pure Python.
    Used by build_signal_prompt to send computed values, never raw OHLCV rows.
    """
    parts = [f"{k}: {v}" for k, v in indicators.items() if v is not None]
    return " | ".join(parts)


import re as _re  # noqa: E402

# ── Layoffs helpers ──────────────────────────────────────────────────────────

STRIP_SUFFIXES: frozenset[str] = frozenset(
    {
        "inc",
        "llc",
        "ltd",
        "corp",
        "co",
        "plc",
        "group",
        "holdings",
        "technologies",
        "tech",
        "corporation",
    }
)

SECTOR_KEYWORDS: dict[str, list[str]] = {
    "tech": ["software", "cloud", "saas", "semiconductor", "chip", "cybersecurity", "fintech"],
    "ai": ["artificial intelligence", "machine learning", "large language model"],
    "finance": ["bank", "banking", "insurance", "capital", "investment", "asset management", "brokerage"],
    "retail": ["retail", "ecommerce", "e-commerce", "consumer goods", "apparel", "fashion"],
    "healthcare": ["health", "pharma", "pharmaceutical", "biotech", "medical", "hospital", "clinical"],
    "media": ["media", "news", "streaming", "entertainment", "publishing", "broadcast", "podcast"],
    "manufacturing": ["manufacturing", "automotive", "auto", "factory", "industrial", "aerospace", "defense"],
    "logistics": ["logistics", "shipping", "delivery", "supply chain", "transportation", "freight"],
}
SECTOR_DEFAULT: str = "tech"

_COMPANY_RE = _re.compile(
    r"(?:Report:\s*|Breaking:\s*|Exclusive:\s*)?([A-Z][^\s]*(?:\s+[A-Z][^\s]*){0,3})'?s?\s+"
    r"(?i:lay(?:s\s+off|off)|cut|slash|shed|trim|reduc|eliminat|terminat|axe|restructur)"
)
_WORKER_RE = _re.compile(r"(\d[\d,]*)\s*(?:employees?|workers?|positions?|jobs?|roles?|people|staff)")


def normalize_company_name(name: str) -> str:
    name = name.lower().strip()
    name = _re.sub(r"[^\w\s]", "", name)
    tokens = [t for t in name.split() if t not in STRIP_SUFFIXES]
    return " ".join(tokens)


def classify_sector(company: str, raw_title: str | None) -> str:
    text = f"{company} {raw_title or ''}".lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        for kw in keywords:
            if _re.search(r"\b" + _re.escape(kw) + r"\b", text):
                return sector
    return SECTOR_DEFAULT


def extract_company_from_title(title: str) -> str:
    m = _COMPANY_RE.match(title.strip())
    return m.group(1).strip() if m else " ".join(title.split()[:5])


def extract_worker_count(text: str) -> int | None:
    m = _WORKER_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def has_minimum_signal(records: list) -> bool:
    return any(r.workers_affected is not None or r.reason is not None for r in records)
