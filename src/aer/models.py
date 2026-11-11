"""Pydantic v2 data models for agent outputs and investment briefs."""

from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aer.risk.models import (
    FactorLoadings,
    WalkForwardResult,
)  # noqa: F401 - re-exported; no circular import (risk/models only imports pydantic)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PortfolioValuationError(Exception):
    """Raised when portfolio valuation produces invalid values (NaN, Inf, negative)."""

    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AgentOutput(BaseModel):
    """Structured output from a single analysis agent."""

    agent_name: str
    content: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)


class InsiderFiling(BaseModel):
    """Single Form 4 open-market purchase filing with signal score."""

    ticker: str
    company_name: str
    insider_name: str
    title: str
    trade_type: str
    filing_date: date
    trade_date: date
    price: float = Field(0.0, ge=0.0)
    quantity: int = Field(0, ge=0)
    value: float = Field(0.0, ge=0.0)
    ownership_change_pct: float = 0.0
    signal_score: float = Field(0.0, ge=0.0)
    signal_label: Literal["STRONG_BUY", "BUY", "WATCH", "NOISE"] = "NOISE"
    filing_lag_days: int = Field(0, ge=0)


class InsiderBuyingOutput(AgentOutput):
    """AgentOutput subclass for insider buying signals from InsiderBuyingAgent."""

    filings: list[InsiderFiling] = Field(default_factory=list)
    scored_tickers: dict[str, float] = Field(default_factory=dict)
    cluster_tickers: list[str] = Field(default_factory=list)
    top_signals: list[InsiderFiling] = Field(default_factory=list)
    as_of: datetime = Field(default_factory=_utcnow)
    data_source: str = "openinsider"
    fetch_errors: list[str] = Field(default_factory=list)


class BacktestResult(BaseModel):
    """Deterministic signal-replay backtest result. Not an AgentOutput - no LLM involved."""

    ticker: str
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float | None = None
    cvar_95: float | None = None
    win_rate: float | None = None
    calmar: float | None = None
    total_return: float | None = None
    period_years: int = 3
    equity_curve_dates: list[str] = Field(default_factory=list)
    equity_curve_values: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Opinion Tracking
# ---------------------------------------------------------------------------


class OpinionRecord(BaseModel):
    """Single agent opinion on a ticker, recorded at a point in time. Frozen - immutable after creation."""

    model_config = ConfigDict(frozen=True)

    ticker: str  # e.g., "AAPL"
    agent_name: str  # e.g., "fundamental"
    opinion_date: date  # Date opinion was recorded
    signal: Literal["BUY", "HOLD", "SELL", "AVOID"]  # Required; no NULL
    confidence: float = Field(ge=0.0, le=1.0)  # Agent confidence; validate on insert: reject NaN
    conviction: float = Field(ge=0.0, le=1.0)  # Brief conviction; validate on insert: reject NaN
    drivers: list[str] = Field(default_factory=list)  # Source tags (e.g., ["strong earnings", "low valuation"])
    logged_at: datetime = Field(default_factory=lambda: datetime.now(UTC))  # When opinion was recorded
    forward_return_5d: float | None = None  # Populated offline; initially NULL

    @field_validator("confidence", "conviction", mode="before")
    @classmethod
    def validate_no_nan_inf(cls, v: Any) -> float:
        """Reject NaN/Inf in confidence/conviction fields.

        Pydantic's ge/le constraints allow NaN to pass (NaN comparisons always return False).
        Explicitly check and reject.
        """
        import math

        if isinstance(v, float):
            if math.isnan(v):
                raise ValueError("confidence/conviction cannot be NaN")
            if math.isinf(v):
                raise ValueError("confidence/conviction cannot be Inf")
        return v


class HitRateMetrics(BaseModel):
    """Aggregated per-agent per-ticker hit-rate metrics from recorded opinions."""

    agent_name: str
    ticker: str
    total_opinions: int = Field(ge=0)
    binary_hit_rate: float | None = Field(default=None, ge=0.0, le=100.0)  # % (0-100); None if insufficient_data
    return_weighted_score: float | None = None  # Decimal (-∞ to +∞, typically [-0.5, +0.5]); None if insufficient_data
    sharpe_ratio: float | None = (
        None  # Annualized (typically -5 to +5); None if insufficient_data or < MIN_VALID_RETURNS_FOR_SHARPE
    )
    conviction_weighted_hit_rate: float | None = Field(
        default=None, ge=0.0, le=100.0
    )  # % (0-100); None if insufficient_data
    n_profitable_holds: int = Field(default=0, ge=0)  # Count of opinions with forward_return > BINARY_HIT_THRESHOLD
    last_opinion_date: date | None = None  # Most recent opinion in sample
    insufficient_data: bool = Field(
        default=False
    )  # True if < MIN_OPINIONS_FOR_METRICS or < MIN_VALID_RETURNS_FOR_SHARPE


# ---------------------------------------------------------------------------
# Options Strategy models
# ---------------------------------------------------------------------------


class SpreadLeg(BaseModel):
    """Single spread leg (short + long side). credit_collected is net per share."""

    short_strike: float
    long_strike: float
    short_delta: float
    long_delta: float
    credit_collected: float
    max_loss_per_contract: float
    expiry: str


class PositionTheta(BaseModel):
    """Theta profile of a single existing options position."""

    ticker: str
    expiry: str
    option_type: Literal["call", "put"]
    strike: float
    contracts: int
    delta: float
    theta_daily: float
    gamma: float
    days_to_expiry: int
    gamma_risk_flag: bool


class RegimeOutput(AgentOutput):
    """Sub-mode 1: IV regime classification and trade-go/no-go verdict."""

    vix_current: float
    vix_regime: Literal["LOW", "NORMAL", "ELEVATED", "CRISIS", "UNKNOWN"]
    iv_rank_proxy: float
    trend_state: Literal["RANGE_BOUND", "TRENDING_UP", "TRENDING_DOWN", "UNKNOWN"]
    iv_edge: Literal["SELLER_EDGE", "NEUTRAL", "BUYER_EDGE"]
    gap_risk: Literal["LOW", "MODERATE", "HIGH"]
    verdict: Literal["GREEN", "YELLOW", "RED"]
    verdict_rationale: str
    macro_event_imminent: bool
    macro_event_name: str
    strategy_recommendation: str


class CreditSpreadOutput(AgentOutput):
    """Sub-mode 2: put/call credit spread recommendation for SPY."""

    underlying: str = "SPY"
    put_spread: SpreadLeg | None = None
    call_spread: SpreadLeg | None = None
    total_credit: float = 0.0
    max_loss: float = 0.0
    breakeven_lower: float = 0.0
    breakeven_upper: float | None = None
    stop_loss_debit: float = 0.0
    profit_target_debit: float = 0.0
    expected_move: float = 0.0
    trade_rationale: str = ""
    skipped: bool = False
    skip_reason: str = ""


class ThetaProjectionOutput(AgentOutput):
    """Sub-mode 3: theta decay projections for existing positions."""

    positions: list[PositionTheta] = Field(default_factory=list)
    daily_income_total: float = 0.0
    gamma_risk_flag: bool = False
    gamma_risk_explanation: str = ""
    projection_30d: float = 0.0
    projection_60d: float = 0.0
    projection_90d: float = 0.0
    projection_assumption: str = "flat position size, no compounding"
    weekend_capture_note: str = ""
    optimal_close_dte: int = 21
    skipped: bool = False
    skip_reason: str = ""


class IronCondorOutput(AgentOutput):
    """Sub-mode 4: iron condor construction with position sizing."""

    underlying: str = ""
    underlying_selection_rationale: str = ""
    put_spread: SpreadLeg | None = None
    call_spread: SpreadLeg | None = None
    total_premium: float = 0.0
    max_loss: float = 0.0
    breakeven_lower: float = 0.0
    breakeven_upper: float = 0.0
    stop_loss_debit: float = 0.0
    profit_target_debit: float = 0.0
    adjustment_trigger_lower: float = 0.0
    adjustment_trigger_upper: float = 0.0
    position_size_contracts: int = 0
    max_risk_dollars: float = 0.0
    account_size_used: float | None = None
    skipped: bool = False
    skip_reason: str = ""


class OptionsStrategyOutput(AgentOutput):
    """Composite output from all four OptionsStrategyAgent sub-modes."""

    regime: RegimeOutput
    credit_spread: CreditSpreadOutput | None = None
    theta_projection: ThetaProjectionOutput | None = None
    iron_condor: IronCondorOutput | None = None
    active_modes: list[str] = Field(default_factory=list)
    suppressed_modes: list[str] = Field(default_factory=list)
    disclaimer: str = ""


# ---------------------------------------------------------------------------
# ML Prediction Layer
# ---------------------------------------------------------------------------


class MLPrediction(BaseModel):
    """Direction classifier output. Frozen - use model_copy() to update."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    direction: Literal["UP", "DOWN"] | None = None
    prob_up_raw: float | None = None  # raw probability; no Platt calibration (see calibrated_prob_up)
    cv_f1_macro: float | None = None
    cv_f1_std: float | None = None
    horizon_days: int = 5
    model_used: Literal["logistic", "lightgbm", "none"] = "lightgbm"
    trained_on_bars: int | None = None
    n_features: int | None = None
    class_balance_up: float | None = None
    brier_score_raw: float | None = None  # evaluated on last CV fold test set
    agrees_with_llm: bool | None = None
    prediction_skipped: bool = False
    backtest_signal: str | None = None  # reserved for arbitration
    backtest_sharpe: float | None = None
    # Ensemble fields - all default None for backward compatibility
    calibrated_prob_up: float | None = None  # LightGBM Platt-calibrated probability
    ensemble_prob_up: float | None = None  # F1-weighted soft vote result
    ensemble_direction: Literal["UP", "DOWN"] | None = None  # derived from ensemble_prob_up
    ml_tension: float | None = None  # abs(prob_up_raw - 0.5) when HOLD; measures ML conviction against neutral LLM
    as_of: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Prediction Markets + Social Intelligence Models
# ---------------------------------------------------------------------------


class PredictionMarketOdds(BaseModel):
    """Single prediction market contract odds (Polymarket or PredictIt)."""

    source: Literal["polymarket", "predictit"]
    contract_title: str
    bid: float = Field(ge=0.0, le=1.0, description="Bid price (probability)")
    ask: float = Field(ge=0.0, le=1.0, description="Ask price (probability)")
    last_price: float = Field(ge=0.0, le=1.0, description="Last traded price")
    timestamp: datetime = Field(default_factory=_utcnow, description="When odds were fetched")
    url: str | None = None

    @field_validator("contract_title")
    @classmethod
    def _validate_contract_title(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("contract_title cannot be empty")
        return v.strip()

    @field_validator("ask")
    @classmethod
    def _validate_ask(cls, v: float, info) -> float:
        data = info.data
        if v < data.get("bid", 0.0):
            raise ValueError("ask must be >= bid (market microstructure)")
        return v

    @field_validator("last_price")
    @classmethod
    def _validate_last_price(cls, v: float, info) -> float:
        data = info.data
        bid = data.get("bid", 0.0)
        ask = data.get("ask", 1.0)
        if not (bid <= v <= ask):
            raise ValueError("last_price must be between bid and ask")
        return v


class SocialSignal(BaseModel):
    """Single social media signal (tweet, post, comment, etc.)."""

    source: Literal["twitter", "discord", "reddit", "stocktwits", "tradingview"]
    content: str = Field(max_length=500, description="Signal text (truncated to 500 chars)")
    content_hash: str = Field(default="", description="SHA-256 hash for dedup (computed externally)")
    sentiment: Literal["bullish", "bearish", "neutral"] = "neutral"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5, description="Signal confidence (0-1)")
    timestamp: datetime = Field(default_factory=_utcnow)
    url: str | None = None

    @field_validator("content")
    @classmethod
    def _validate_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content cannot be empty")
        return v.strip()


class SocialIntelligence(BaseModel):
    """Aggregated social signals for a ticker (Layer 1 output + optional Layer 2 synthesis)."""

    ticker: str
    signals: list[SocialSignal] = Field(default_factory=list)
    bullish_count: int = Field(ge=0, default=0)
    bearish_count: int = Field(ge=0, default=0)
    neutral_count: int = Field(ge=0, default=0)
    top_keywords: list[str] = Field(default_factory=list, description="Top N extracted keywords")
    source_breakdown: dict[str, int] = Field(default_factory=dict, description="Count per source")
    layer2_synthesis: str | None = Field(default=None, max_length=200, description="Optional Claude Haiku summary")
    timestamp: datetime = Field(default_factory=_utcnow)

    @field_validator("ticker")
    @classmethod
    def _validate_ticker(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ticker cannot be empty")
        return v.upper().strip()

    @field_validator("bullish_count", "bearish_count", "neutral_count", mode="after")
    @classmethod
    def _validate_signal_counts(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError("Signal counts must be non-negative")
        return v

    @field_validator("layer2_synthesis")
    @classmethod
    def _validate_layer2_synthesis(cls, v: str | None) -> str | None:
        if v and not v.strip():
            return None
        return v


class InvestmentBrief(BaseModel):
    """Full investment brief compiled from all agent outputs."""

    ticker: str
    company_name: str
    exchange: str = ""
    sector: str = ""
    date: datetime = Field(default_factory=_utcnow)

    current_price: float | None = None
    price_target: float | None = None
    price_currency: str = "USD"

    recommendation: Literal["BUY", "HOLD", "SELL", "AVOID"] = "HOLD"
    conviction: float = Field(ge=0.0, le=1.0, default=0.5, description="0=low, 1=high")
    summary: str = ""

    fundamental: AgentOutput | None = None
    technical: AgentOutput | None = None
    sentiment: AgentOutput | None = None
    macro: AgentOutput | None = None
    positioning: AgentOutput | None = None
    news: AgentOutput | None = None
    insider: InsiderBuyingOutput | None = None

    risks: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    time_horizon: Literal["short", "medium", "long"] = "medium"
    backtesting: BacktestResult | None = None
    options_strategy: OptionsStrategyOutput | None = None
    ml_prediction: MLPrediction | None = None
    factor_loadings: FactorLoadings | None = None

    # Computed price targets (ATR-based, BUY/SELL recommendations only)
    entry_price: float | None = None
    entry_method: str | None = None  # "support_level" | "atr_buffer" | "fallback_pct"
    target_price: float | None = None  # computed ATR-based exit (distinct from price_target LLM field)
    target_method: str | None = None  # "risk_reward_2_1" | "resistance" | "fallback_pct"
    stop_loss_price: float | None = None
    risk_reward_ratio: float | None = None

    # Prefetch completeness fields.
    # data_freshness maps each domain name to its status string:
    #   "fetched"          - cache was cold; prefetch warmed it this run
    #   "cache-hit"        - cache was already warm at prefetch time
    #   "skipped-by-budget" - domain hit its per-domain call budget (NaN-vs-zero: NOT a true miss)
    #   "errored"          - fetcher failed or timed out
    #   None               - not populated (brief built without prefetch run)
    # degraded=True signals that at least one domain was skipped-by-budget or errored,
    # so the brief may be partially cold. Distinguishes missing-by-budget from true-zero.
    data_freshness: dict[str, str] | None = None
    degraded: bool = False

    # Walk-forward optimization results (diagnostic, does not influence recommendation)
    walk_forward: WalkForwardResult | None = None

    # Vintage timestamps - Unix timestamps (float) for each data type (prices_ts, consensus_ts, macro_ts, etc.)
    vintage: dict[str, float | None] = Field(default_factory=dict)

    # Prediction Markets (Polymarket + PredictIt odds)
    prediction_markets: list["PredictionMarketOdds"] = Field(default_factory=list)

    # Social Intelligence (5-source signals + optional Layer 2 synthesis)
    social_intelligence: "SocialIntelligence | None" = None

    def get_vix_regime(self) -> str | None:
        """Return VIX regime from options_strategy (typed) or macro metadata (fallback)."""
        if self.options_strategy is not None:
            return self.options_strategy.regime.vix_regime
        if self.macro is not None:
            return self.macro.metadata.get("vix_regime")
        return None


# ---------------------------------------------------------------------------
# Synthesis structured output (used by graph.py synthesis node)
# ---------------------------------------------------------------------------


class SynthesisOutput(BaseModel):
    recommendation: Literal["BUY", "HOLD", "SELL", "AVOID"]
    conviction: float = Field(ge=0.0, le=1.0, description="0=low conviction, 1=high conviction")
    price_target: float | None = Field(
        default=None, description="12-month price target in USD, or null if not determinable"
    )
    time_horizon: Literal["short", "medium", "long"] = "medium"
    summary: str = Field(default="", description="2-3 sentence investment thesis")
    risks: list[str] = Field(default_factory=list, description="3-5 key risks, each one sentence")
    catalysts: list[str] = Field(default_factory=list, description="3-5 key catalysts, each one sentence")

    @field_validator("risks", "catalysts", mode="before")
    @classmethod
    def _flatten_str_list(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [" ".join(str(x) for x in item) if isinstance(item, list) else str(item) for item in v]
