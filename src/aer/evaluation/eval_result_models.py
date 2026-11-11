"""Typed Pydantic v2 result contract for the alt-data eval framework.

Import-cheap: only pydantic, enum, and typing are imported.
No heavy data or agent dependencies - test collection does not pull yfinance.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class GateResult(str, Enum):
    """Gate outcome for a layoffs precision evaluation run."""

    EVALUATED = "evaluated"  # N >= EVAL_MIN_N_BEARISH; precision computed and compared to gate
    INSUFFICIENT_DATA = "insufficient_data"  # N < EVAL_MIN_N_BEARISH; no point estimate produced
    DATA_UNAVAILABLE = "data_unavailable"  # infra failure (skip ratio exceeded threshold)


class EvalStatus(str, Enum):
    """Evaluation status for agents that cannot produce a precision metric."""

    NOT_EVALUABLE = "not_evaluable"  # instagram emits unstructured free-text; no precision possible


class FrozenPriceRecord(BaseModel):
    """Per-record price tuple frozen into the baseline for offline precision recomputation.

    Precision must be recomputable from the baseline JSON alone,
    without re-fetching price data.
    """

    # frozen=True: this record is the offline-recomputation substrate and must be
    # immutable. Constructed in a single call, so frozen never raises.
    model_config = {"frozen": True}

    ticker: str
    entry_date: str  # ISO-8601 date string (T+1 entry, no same-day)
    exit_date: str  # ISO-8601 date string (T+11 exit)
    entry_close: float
    exit_close: float
    excess: float  # instrument_fwd_return - SPY_fwd_return over the same window


class LayoffsEvalResult(BaseModel):
    """Typed result of a layoffs agent precision evaluation run.

    Gate fields (gate_result, n_bearish_calls, n_bearish_hits, gate_precision, passed_gate)
    encode the sole gate metric. All other metric fields are non-gating diagnostics.
    Diagnostic fields default to None so a first-run insufficient_data result is constructible.
    """

    model_config = {"frozen": False}

    # Identity and protocol provenance
    agent: Literal["layoffs"]
    eval_date: str  # ISO-8601 date string; computed inside run_eval(), never at import time
    protocol_version: str  # echoes EVAL_PROTOCOL_VERSION
    forward_window_days: int  # echoes EVAL_FORWARD_WINDOW_DAYS
    min_precision_gate: float  # echoes EVAL_MIN_PRECISION_GATE
    min_n_bearish: int  # echoes EVAL_MIN_N_BEARISH

    # Gate fields (the sole pass/fail decision surface)
    gate_result: GateResult
    n_bearish_calls: int  # resolved BEARISH records included in gate computation
    n_bearish_hits: int  # BEARISH calls where ETF excess < 0
    gate_precision: float | None = None  # None when gate_result != EVALUATED
    passed_gate: bool | None = None  # None when gate_result != EVALUATED

    # Non-gating diagnostic metrics (reported; never alter gate_result/passed_gate/N)
    recall: float | None = None  # BEARISH calls / total_evaluable underperformers
    average_precision: float | None = None  # average_precision_score (pseudo-scores: BEARISH=1/CAUTIOUS=0.5/NEUTRAL=0)
    # mean(ETF_fwd_return - SPY_fwd_return) over all evaluable resolved records (BEARISH+CAUTIOUS+NEUTRAL)
    diagnostic_mean_excess: float | None = None
    diagnostic_deadband_precision: float | None = None  # precision with vol-scaled deadband
    # Diagnostic A: Wilson 95% score interval on gate_precision
    gate_precision_ci_low: float | None = None
    gate_precision_ci_high: float | None = None
    # Diagnostic B: base-rate null and lift
    base_rate: float | None = None  # fraction of all evaluable records where excess < 0
    diagnostic_precision_lift: float | None = None  # gate_precision - base_rate
    # Diagnostic C: OLS beta-adjusted excess precision
    diagnostic_beta_adjusted_precision: float | None = None
    beta_hat_mean: float | None = None
    beta_hat_min: float | None = None
    beta_hat_max: float | None = None
    # Diagnostic D: placebo control (seeded run_date shuffle)
    diagnostic_placebo_precision: float | None = None

    # Counters (all resolved and skipped record bookkeeping)
    total_records_read: int = 0
    # signal in EVAL_EVALUABLE_SIGNALS, sector mapped, exit date resolved (reaches price fetch)
    total_evaluable: int = 0
    skipped_no_price_data: int = 0
    skipped_malformed: int = 0  # bad JSON, missing signal, non-ISO run_date
    skipped_unmapped_sector: int = 0  # sector absent from SECTOR_TO_ETF
    pending_resolution: int = 0  # exit date >= today or in the future

    # Provenance and offline reproducibility
    jsonl_files_read: list[str] = []  # paths of all JSONL files consumed in this run
    frozen_price_records: list[FrozenPriceRecord] = []


class InstagramEvalStatus(BaseModel):
    """Evaluation status for the instagram agent (not evaluable at present).

    Instagram emits only free-text insights (no ticker, no direction).
    A precision metric would measure the extractor, not the agent - so we document
    the status honestly rather than producing a misleading number.
    """

    model_config = {"frozen": False}

    agent: Literal["instagram"]
    eval_date: str  # ISO-8601 date string
    status: EvalStatus  # always NOT_EVALUABLE until IG emits structured output
    reason: str  # human-readable rationale (e.g. "unstructured signal: no ticker or direction")
    total_records_read: int = 0
    jsonl_files_read: list[str] = []
