import functools
import subprocess
from pathlib import Path
from typing import Final

# Schema
SCHEMA_VERSION: Final[int] = 2

# --- I/O caps and subprocess timeouts ---
WATCHLIST_MAX_TICKERS: Final[int] = 50
"""Hard cap applied by watchlist load AND save (src/aer/utils/watchlist.py) — bounds pipeline
fan-out and LLM cost per run. Boundary-tested at 49/50/51."""

GIT_TIMEOUT_SECONDS: Final[int] = 5
"""Timeout for git subprocess calls (rev-parse); git on a local repo is sub-second."""


PIPELINE_LOCK_STALE_HOURS: Final[int] = 6
"""A pipeline/simulator lock older than this (or owned by a dead PID) is reclaimable."""

PIPELINE_LOG_PATH: Final[Path] = Path("logs") / "run_pipeline.log"
"""Repo-relative on-disk log for run_pipeline.py's __main__ entry point. First
log-path constant in this module; precedent for any sibling log-path constants."""


SYNTHESIS_MODEL: Final[str] = "claude-sonnet-4-6"


# Agent versioning - short git SHA, resolved lazily on first access.
# Importing this module no longer spawns a subprocess; `AGENT_VERSION` is served by the
# module-level __getattr__ below (PEP 562), so `from aer.utils.constants import AGENT_VERSION`
# keeps working and pays the lookup once, at that import, not at every import of this module.
@functools.lru_cache(maxsize=1)
def _get_agent_version() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=GIT_TIMEOUT_SECONDS,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        # OSError: git binary absent or cwd unreadable. SubprocessError: non-zero exit
        # (not a repo) or TimeoutExpired. Anything else is a real bug and propagates.
        return "unknown"
    return sha or "unknown"


def __getattr__(name: str) -> str:
    if name == "AGENT_VERSION":
        return _get_agent_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Regime
REGIME_ADX_TRENDING: Final[int] = 20
REGIME_ADX_VOLATILE: Final[int] = 25
REGIME_ATR_VOLATILE_PCT: Final[float] = 1.5
REGIME_ATR_WINDOW: Final[int] = 14
REGIME_ADX_WINDOW: Final[int] = 14

# Suppression
SUPPRESSION_MIN_SAMPLES: Final[int] = 20
SUPPRESSION_HIT_RATE_FLOOR: Final[float] = 0.45
SUPPRESSION_REVIEW_WINDOW_DAYS: Final[int] = 30


# CUSUM - target is coin-flip baseline, achievable in any regime
CUSUM_TARGET_HIT_RATE: Final[float] = 0.50
CUSUM_SLACK: Final[float] = 0.02
CUSUM_THRESHOLD: Final[float] = 5.0

# Data quality
BAR_RESOLUTION_MINUTES: Final[int] = 5
SESSION_BAR_COUNT: Final[int] = 78  # RTH 9:30–16:00 at 5-min resolution
DATA_MAX_STALENESS_MINUTES: Final[int] = 15
DATA_MIN_BAR_COMPLETENESS: Final[float] = 0.98  # 0.98 × 78 = 76.4 bars minimum
OUTCOME_RETURN_PLAUSIBILITY_CAP: Final[float] = 0.30  # |return| > 30% on D+1 is implausible


# Reasoning
REASONING_SUMMARY_MAX_CHARS: Final[int] = 200


# Conflict classifier
CONFLICT_REPORT_THRESHOLD: Final[int] = 6
CONF_MACRO_OVERRIDE_CONFIDENCE_FLOOR: Final[float] = 0.6
CONF_HIGH_CONVICTION_MIN_AGENTS: Final[int] = 3
RISK_SCORE_WARNING_THRESHOLD: Final[int] = 7
SENTIMENT_NEGATIVE_THRESHOLD: Final[float] = -0.5
# Conflict pattern name constants (used as flag strings in professor_flags JSON arrays)
# All 8 patterns emitted by conflict_classifier.py - values must match exactly
CONF_HIGH_CONVICTION_HIGH_RISK: Final[str] = "CONF_HIGH_CONVICTION_HIGH_RISK"
CONF_MACRO_OVERRIDE: Final[str] = "CONF_MACRO_OVERRIDE"
CONF_SMART_MONEY_FLOW: Final[str] = "CONF_SMART_MONEY_FLOW"
CONF_SENTIMENT_DIVERGENCE: Final[str] = "CONF_SENTIMENT_DIVERGENCE"
CONF_TECHNICAL_REVERSAL: Final[str] = "CONF_TECHNICAL_REVERSAL"
CONF_REGIME_MISMATCH: Final[str] = "CONF_REGIME_MISMATCH"
CONF_CONVICTION_SPLIT: Final[str] = "CONF_CONVICTION_SPLIT"
CONF_ML_SIGNAL_MISMATCH: Final[str] = "CONF_ML_SIGNAL_MISMATCH"

# Regime-to-template-tag mapping
REGIME_TO_TEMPLATE_TAG: Final[dict[str, str]] = {
    "BULL_STEADY": "growth",
    "BULL_VOLATILE": "volatile",
    "BEAR": "risk",
    "STAGFLATION": "risk",
    "RECOVERY": "growth",
    "UNKNOWN": "neutral",
}


# Technical-signal thresholds (agents/technical.py::_classify_signals).
# Bare literals here drive BUY/SELL/HOLD sub-signals; each must have a boundary test.
TECH_ADX_TREND_MIN: Final[int] = 25  # ADX >= this confirms a trend (trend_signal)
TECH_MOMENTUM_MIDLINE: Final[float] = 50.0  # RSI / Stoch %K midline (momentum_signal)
TECH_MOMENTUM_MAJORITY_FRAC: Final[float] = 0.75  # fraction of present indicators agreeing → directional
TECH_MOMENTUM_MIN_AGREEING: Final[int] = 3  # absolute count for strong agreement (max(N, n))
TECH_STOCH_OVERSOLD: Final[int] = 20  # Stoch %K < this → mean-reversion BULLISH
TECH_STOCH_OVERBOUGHT: Final[int] = 80  # Stoch %K > this → mean-reversion BEARISH
TECH_ATR_HIGH_VOL_PCT: Final[float] = 3.0  # ATR/price*100 > this → HIGH volatility
TECH_ATR_LOW_VOL_PCT: Final[float] = 1.0  # ATR/price*100 < this → LOW volatility
TECH_BB_WIDTH_HIGH_PCT: Final[float] = 5.0  # BB width/price*100 > this → HIGH volatility
TECH_OVERALL_CONSENSUS_MIN: Final[int] = 2  # sub-signals agreeing (of 3) → overall direction

# Net liquidity (macro.py::get_net_liquidity / _align_net_liquidity)
# Week-over-week net-liquidity delta threshold in billions USD separating expanding/contracting from neutral.
NET_LIQUIDITY_TREND_THRESHOLD_BN: Final[float] = 50.0
# Observations fetched for the DAILY RRP series so .asof(prev_anchor) reaches the previous weekly WALCL
# date (~7 calendar days back); 4 daily points cannot span a week, which would make net_liquidity_trend
# perpetually None. 15 business days ≈ 3 weeks gives margin for holidays.
NET_LIQUIDITY_DAILY_LOOKBACK_PERIODS: Final[int] = 15

# Alt-data eval - locked protocol constants
# Gate fence: these values are pre-committed and must not change without unlocking the decision.
EVAL_FORWARD_WINDOW_DAYS: Final[int] = 10  # T+1 entry → T+11 exit (10 trading days)
EVAL_MIN_PRECISION_GATE: Final[float] = 0.55  # gate threshold (BEARISH precision ≥ 55%)
EVAL_MIN_N_BEARISH: Final[int] = 30  # minimum BEARISH observations required to evaluate gate
EVAL_PROTOCOL_VERSION: Final[str] = "1.3"  # single canonical version echoed into baseline JSON


# Alt-data eval: sector→SPDR ETF mapping, aligned to the classify_sector taxonomy
# Keys that classify_sector can emit: tech, ai, finance, healthcare, retail, media, manufacturing, logistics
# Additional ETF-only keys retained for forward-compatibility (never emitted by classify_sector currently)
SECTOR_TO_ETF: Final[dict[str, str]] = {
    "tech": "XLK",
    "ai": "XLK",  # explicit: 'ai' must not fall through to KeyError
    "finance": "XLF",
    "healthcare": "XLV",
    "retail": "XLY",
    "media": "XLC",
    "manufacturing": "XLI",
    "logistics": "XLI",
    "energy": "XLE",  # not currently emitted by classify_sector
    "staples": "XLP",  # not currently emitted by classify_sector
    "materials": "XLB",  # not currently emitted by classify_sector
    "real_estate": "XLRE",  # not currently emitted by classify_sector
    "utilities": "XLU",  # not currently emitted by classify_sector
}

# Alt-data eval: evaluable signal classes (NEUTRAL kept for recall universe; INSUFFICIENT_DATA excluded)
EVAL_EVALUABLE_SIGNALS: Final[frozenset[str]] = frozenset({"BEARISH", "CAUTIOUS", "NEUTRAL"})

# Prefetch - per-domain external-call budgets.
# A budget bounds external calls (misses) for that domain in one daily pass.
# Per-ticker domains scale with watchlist size; global domains expect ~1.
# [ASSUMED] sizing: the prefetch pass derives per-domain budgets at runtime from
# len(load_watchlist()) × fanout × margin and asserts these statics are >= the realized cold-pass need.
PREFETCH_BUDGET_PRICES: Final[int] = 30
PREFETCH_BUDGET_FINANCIALS: Final[int] = 20
PREFETCH_BUDGET_EARNINGS: Final[int] = 20
PREFETCH_BUDGET_OWNERSHIP: Final[int] = 15
PREFETCH_BUDGET_OPTIONS: Final[int] = 15
PREFETCH_BUDGET_MACRO: Final[int] = 10
PREFETCH_BUDGET_BREADTH: Final[int] = 3
PREFETCH_BUDGET_REGIME: Final[int] = 3
PREFETCH_BUDGET_SENTIMENT: Final[int] = 40
PREFETCH_BUDGET_NEWS: Final[int] = 40
# Global budget for Ken French daily factor data fetch (one global download per run).
PREFETCH_BUDGET_FACTORS: Final[int] = 1

# Social Intelligence + Cost Control
MAX_TWITTER_COST_PER_DAY_USD: Final[float] = 5.00  # $5/day budget for X API

# Prefetch data-quality - market-aware / stress TTL + corporate-action / earnings windows
# Short TTL for prices/quote during market hours. Must be < TTL_PRICES / <= TTL_QUOTE.
# [ASSUMED] sizing - tunable. 15 min covers one intraday candle; 5 min keeps quotes near-live.
PREFETCH_MARKET_HOURS_TTL_PRICES: Final[int] = 900  # 15 min - [ASSUMED] short market-hours prices TTL
PREFETCH_MARKET_HOURS_TTL_QUOTE: Final[int] = 300  # 5 min - [ASSUMED] short market-hours quote TTL
# Number of trading days on each side of next-earnings date that triggers force-refresh.
PREFETCH_EARNINGS_WINDOW_TRADING_DAYS: Final[int] = 1  # ±1 trading day
# Number of trading days on or after ex-dividend date that invalidates adjusted OHLCV.
# 0 = same-day ex-date invalidation only (today is ex-date → purge; yesterday is not covered).
PREFETCH_EXDATE_WINDOW_TRADING_DAYS: Final[int] = 0  # 0 = today only
# Integer divisor applied to a domain's base TTL on a stress regime. Must be > 1 so
# stress shortens TTL, never lengthens it. [ASSUMED] value of 4 (e.g. 24h → 6h on stress).
PREFETCH_STRESS_TTL_DIVISOR: Final[int] = 4  # base_ttl // 4 on stress regime

# ---------------------------------------------------------------------------
# Fama-French factor loadings
# ---------------------------------------------------------------------------
FACTOR_ROLLING_WINDOW_MONTHS: Final[int] = 60  # 5-year OLS window
FACTOR_PREFERRED_MONTHS: Final[int] = 60  # full window preferred; < this → low_data_quality
FACTOR_MIN_MONTHS: Final[int] = 36  # revised: 36-month soft floor; < this → insufficient_history


# ---------------------------------------------------------------------------
# Scaffold constants
# ---------------------------------------------------------------------------
REQUIRED_SEVERITY_KEY: Final[str] = "severity"

# ---------------------------------------------------------------------------
# Trading Signal Thresholds
# All decision boundaries for technical signal classification. No bare literals in
# technical.py or python_replacements.py - all comparisons use these named constants.
# ---------------------------------------------------------------------------
RSI_OVERBOUGHT: Final[float] = 70.0
"""RSI >= 70 indicates overbought condition; typically a sell signal."""

RSI_OVERSOLD: Final[float] = 30.0
"""RSI <= 30 indicates oversold condition; typically a buy signal."""

MACD_SIGNAL_THRESHOLD: Final[float] = 0.0
"""MACD line crossing zero indicates momentum shift; positive = bullish, negative = bearish."""

BBL_BREAKOUT_THRESHOLD: Final[float] = 0.95
"""Price must exceed upper Bollinger Band at 95% level to signal breakout (avoids false positives)."""

VOLUME_MA_THRESHOLD: Final[float] = 1.2
"""Volume spike: current volume / 20-day MA must exceed 1.2 to confirm signal."""

ATR_VOLATILITY_HIGH: Final[float] = 3.0
"""ATR > 3.0 indicates high volatility environment; affects position sizing and risk management."""

MOMENTUM_THRESHOLD_WEAK: Final[float] = 0.75
"""Momentum signal below 0.75 is considered weak; above 0.75 is strong."""

ICHIMOKU_SIGNAL_THRESHOLD: Final[float] = 0.5
"""Ichimoku signal: price must clear signal line + 0.5 to confirm trend."""

# ---------------------------------------------------------------------------
# Position Sizing Policy: canonical thresholds for all sizing decisions
# ---------------------------------------------------------------------------
MAX_POSITIONS: Final[int] = 10
"""Maximum concurrent open positions (policy constant)."""

MAX_POSITION_PCT: Final[float] = 0.05
"""Maximum position size per signal (5% of portfolio; policy constant)."""

KELLY_FRACTION: Final[float] = 0.25
"""Kelly criterion scale factor (quarter-Kelly for conservative sizing)."""

# ---------------------------------------------------------------------------
# Daily Simulator Configuration: conforms to the policy constants above
# ---------------------------------------------------------------------------
SIMULATOR_FRACTIONAL_SHARE_DECIMALS: Final[int] = (
    6  # Decimal precision for share counts (6 decimals = 1 millionth of a share; sufficient for positions up to $1M at $1/share scale)
)
SIMULATOR_INITIAL_CASH: Final[float] = 1000.0  # Starting portfolio value in USD


def validate_configuration() -> None:
    """Validate policy and simulator configuration thresholds. Raises ValueError on impossible values.

    Called at module import time to catch configuration errors early.
    """
    errors = []

    # Policy constants
    if MAX_POSITIONS <= 0:
        errors.append(f"MAX_POSITIONS must be > 0, got {MAX_POSITIONS}")
    if MAX_POSITION_PCT <= 0.0 or MAX_POSITION_PCT >= 1.0:
        errors.append(f"MAX_POSITION_PCT must be (0, 1), got {MAX_POSITION_PCT}")
    if KELLY_FRACTION <= 0.0 or KELLY_FRACTION > 1.0:
        errors.append(f"KELLY_FRACTION must be (0, 1], got {KELLY_FRACTION}")

    # Simulator configuration
    if SIMULATOR_FRACTIONAL_SHARE_DECIMALS < 0:
        errors.append(f"SIMULATOR_FRACTIONAL_SHARE_DECIMALS must be >= 0, got {SIMULATOR_FRACTIONAL_SHARE_DECIMALS}")

    if SIMULATOR_INITIAL_CASH <= 0:
        errors.append(f"SIMULATOR_INITIAL_CASH must be > 0, got {SIMULATOR_INITIAL_CASH}")

    if errors:
        raise ValueError("Configuration validation failed:\n  " + "\n  ".join(errors))


# Validate on import
validate_configuration()

# ---------------------------------------------------------------------------
# Decision-gate review constants
# ---------------------------------------------------------------------------
# Freshness thresholds and measurement constants for decision-gate review.
# NOTE: Position sizing thresholds must reference SIMULATOR_* constants above - not duplicated here.


GATE_G5_OLLAMA_UPTIME_FLOOR: Final[float] = 0.95
"""Ollama fallback health (distinct from KILL_SWITCH_OLLAMA_UPTIME_FLOOR, the separate 0.80 cost floor
in cost_thresholds.py). 7-day uptime < 95% or zero uptime data = gate FAIL-BY-ABSENCE."""

# Position sizing: the daily simulator sizes via
# min(MAX_POSITION_PCT, cash/equity) x equity using MAX_POSITIONS and MAX_POSITION_PCT above.
# Do not duplicate these constants (single source of truth).

# Free-tier budgets for the LLM usage meter (src/aer/utils/usage_meter.py). Groq free plan, openai/gpt-oss-120b,
# console.groq.com/docs/rate-limits read 2026-09-28. A process-level WARNING fires once per provider when
# its metered tokens reach USAGE_BUDGET_WARN_FRACTION of the daily budget.
GROQ_FREE_TOKENS_PER_DAY: Final[int] = 200_000
GROQ_FREE_REQUESTS_PER_DAY: Final[int] = 1_000
USAGE_BUDGET_WARN_FRACTION: Final[float] = 0.8
