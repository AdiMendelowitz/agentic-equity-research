from pathlib import Path

import pytest
from pydantic import ValidationError

from aer.evaluation import weekly_scorer
from aer.evaluation.spec import DEFAULT_SPEC, EvaluationSpec, is_correct
from aer.ml.constants import PREDICTION_HORIZON_DAYS


def test_default_horizon_matches_constant() -> None:
    assert DEFAULT_SPEC.horizon_days == PREDICTION_HORIZON_DAYS
    assert DEFAULT_SPEC.resolution_lag_trading_days == PREDICTION_HORIZON_DAYS


def test_spec_is_frozen() -> None:
    with pytest.raises(ValidationError):
        DEFAULT_SPEC.horizon_days = 3  # type: ignore[misc]


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        EvaluationSpec(unexpected=1)  # type: ignore[call-arg]


def test_zero_horizon_rejected() -> None:
    with pytest.raises(ValidationError):
        EvaluationSpec(horizon_days=0, resolution_lag_trading_days=1)


def test_resolution_lag_below_horizon_rejected() -> None:
    with pytest.raises(ValidationError):
        EvaluationSpec(horizon_days=5, resolution_lag_trading_days=4)


def test_baselines_contain_required_entries() -> None:
    assert "always_hold" in DEFAULT_SPEC.baselines
    assert "shuffled_label" in DEFAULT_SPEC.baselines


def test_weekly_scorer_references_default_spec() -> None:
    source = Path(weekly_scorer.__file__).read_text(encoding="utf-8")
    assert "DEFAULT_SPEC" in source


def test_every_class_has_a_rule() -> None:
    assert set(DEFAULT_SPEC.resolution_rules) == {"BUY", "SELL", "AVOID", "HOLD"}


def test_missing_class_rejected() -> None:
    rules = dict(DEFAULT_SPEC.resolution_rules)
    del rules["HOLD"]
    with pytest.raises(ValidationError):
        EvaluationSpec(resolution_rules=rules)


def test_extra_class_rejected() -> None:
    rules = {**DEFAULT_SPEC.resolution_rules, "WATCH": "forward_return_above_cost"}
    with pytest.raises(ValidationError):
        EvaluationSpec(resolution_rules=rules)


@pytest.mark.parametrize(
    ("recommendation", "forward_return", "expected"),
    [
        ("BUY", 0.002, True),
        ("BUY", 0.0005, False),
        ("BUY", 0.001, False),
        ("BUY", -0.002, False),
        ("SELL", -0.002, True),
        ("SELL", -0.0005, False),
        ("SELL", -0.001, False),
        ("SELL", 0.002, False),
        ("AVOID", -0.002, True),
        ("AVOID", 0.002, False),
        ("AVOID", -0.001, False),
        ("HOLD", 0.0005, True),
        ("HOLD", 0.001, True),
        ("HOLD", -0.001, True),
        ("HOLD", 0.002, False),
        ("HOLD", -0.002, False),
    ],
)
def test_is_correct_boundaries(recommendation: str, forward_return: float, expected: bool) -> None:
    assert is_correct(recommendation, forward_return, 0.001) is expected


def test_is_correct_unknown_recommendation() -> None:
    with pytest.raises(KeyError):
        is_correct("WATCH", 0.0, 0.001)
