"""Evaluation data models and schema for outcome tracking.

Note on synth_correct horizon mismatch:
The LLM recommendation is a weeks-to-months thesis; synth_correct evaluates it against
5-day price movement. This comparison is structurally biased against the LLM - it is
logged for observability only. Do NOT use it to trigger model replacement decisions
(should_trigger_review intentionally excludes this comparison from replacement logic).
"""

import time
from dataclasses import dataclass, field
from typing import Literal

EVAL_DB_SCHEMA_VERSION: int = 1

EVAL_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS agent_eval (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    brief_date          TEXT    NOT NULL,
    recommendation      TEXT    NOT NULL,
    conviction          REAL    NOT NULL,
    ml_direction        TEXT,
    ml_prob_up_raw      REAL,
    ml_cv_f1_macro      REAL,
    ml_cv_f1_std        REAL,
    ml_model_used       TEXT,
    ml_prediction_skipped INTEGER NOT NULL DEFAULT 0,
    backtest_signal     TEXT,
    backtest_sharpe     REAL,
    vix_regime          TEXT,
    structural_quality  TEXT,
    resolution_date     TEXT,
    actual_direction    TEXT,
    actual_5d_return    REAL,
    synth_correct       INTEGER,
    ml_correct          INTEGER,
    resolution_attempts INTEGER NOT NULL DEFAULT 0,
    resolution_status   TEXT    NOT NULL DEFAULT 'PENDING',
    created_at          REAL    NOT NULL,
    updated_at          REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_ticker_date ON agent_eval(ticker, brief_date);
CREATE INDEX IF NOT EXISTS idx_eval_resolution ON agent_eval(resolution_date, resolution_status)
"""


@dataclass(frozen=True)
class AgentEvalRecord:
    """Immutable snapshot of a brief evaluation record."""

    ticker: str
    brief_date: str
    recommendation: str
    conviction: float
    ml_direction: str | None = None
    ml_prob_up_raw: float | None = None
    ml_cv_f1_macro: float | None = None
    ml_cv_f1_std: float | None = None
    ml_model_used: str | None = None
    ml_prediction_skipped: bool = False
    backtest_signal: str | None = None
    backtest_sharpe: float | None = None
    vix_regime: str | None = None
    structural_quality: str | None = None  # JSON string of dict[str, bool]
    resolution_date: str | None = None
    actual_direction: str | None = None
    actual_5d_return: float | None = None
    synth_correct: bool | None = None  # None for HOLD - never False (see module docstring)
    ml_correct: bool | None = None
    resolution_attempts: int = 0
    resolution_status: Literal["PENDING", "RESOLVED", "UNRESOLVABLE"] = "PENDING"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def compute_synth_correct(recommendation: str, actual_direction: str) -> bool | None:
    """Map LLM recommendation + actual direction to correctness signal.

    HOLD → None: abstention, never scored as wrong.
    BUY → correct if actual_direction == "UP".
    SELL or AVOID → correct if actual_direction == "DOWN".
    """
    if recommendation == "HOLD":
        return None
    if recommendation == "BUY":
        return actual_direction == "UP"
    if recommendation in ("SELL", "AVOID"):
        return actual_direction == "DOWN"
    return None
