"""The single frozen definition of what the evaluation scores, shared by the scorer and the report.

The per-recommendation resolution rules apply to synthetic demo output; the published whole-sample aggregates were
not re-scored with them, because the HOLD rule was defined after those aggregates were seen.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aer.ml.constants import PREDICTION_HORIZON_DAYS

ResolutionRule = Literal[
    "forward_return_above_cost",
    "forward_return_below_minus_cost",
    "absolute_forward_return_within_cost",
]

_RECOMMENDATION_CLASSES = frozenset({"BUY", "SELL", "AVOID", "HOLD"})


def _default_resolution_rules() -> dict[str, ResolutionRule]:
    return {
        "BUY": "forward_return_above_cost",
        "SELL": "forward_return_below_minus_cost",
        "AVOID": "forward_return_below_minus_cost",
        "HOLD": "absolute_forward_return_within_cost",
    }


class EvaluationSpec(BaseModel):
    """Frozen evaluation definition. The resolution_rules mapping must be treated as read-only after construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon_days: int = PREDICTION_HORIZON_DAYS
    label_rule: Literal["close_to_close_return_above_round_trip_cost"] = "close_to_close_return_above_round_trip_cost"
    resolution_lag_trading_days: int = PREDICTION_HORIZON_DAYS
    baselines: tuple[str, ...] = Field(default=("always_hold", "shuffled_label"))
    resolution_rules: dict[str, ResolutionRule] = Field(default_factory=_default_resolution_rules)

    @model_validator(mode="after")
    def _check_consistency(self) -> "EvaluationSpec":
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be at least 1")
        if self.resolution_lag_trading_days < self.horizon_days:
            raise ValueError("resolution_lag_trading_days must not be shorter than horizon_days")
        if set(self.resolution_rules) != _RECOMMENDATION_CLASSES:
            raise ValueError(f"resolution_rules keys must be exactly {sorted(_RECOMMENDATION_CLASSES)}")
        return self


DEFAULT_SPEC = EvaluationSpec()


def is_correct(recommendation: str, forward_return: float, cost: float, spec: EvaluationSpec = DEFAULT_SPEC) -> bool:
    rule = spec.resolution_rules[recommendation]
    if rule == "forward_return_above_cost":
        return forward_return > cost
    if rule == "forward_return_below_minus_cost":
        return forward_return < -cost
    return abs(forward_return) <= cost
