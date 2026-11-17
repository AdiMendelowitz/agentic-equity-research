"""The brief carries the prediction of the model the gate validated, chosen by the same rule as fit_ensemble."""

from typing import Literal

import pytest

from aer.demo.generate import synthetic_series
from aer.ml.features import build_feature_matrix
from aer.ml.models import fit_ensemble, selected_prediction
from aer.models import MLPrediction


def _pred(model_used: Literal["lightgbm", "logistic"], f1: float | None, prob_up: float) -> MLPrediction:
    return MLPrediction(
        ticker="SYN",
        direction="UP" if prob_up >= 0.5 else "DOWN",
        prob_up_raw=prob_up,
        cv_f1_macro=f1,
        model_used=model_used,
        trained_on_bars=500,
        n_features=10,
    )


def test_logistic_is_scored_when_it_wins_the_gate() -> None:
    chosen = selected_prediction(_pred("lightgbm", 0.51, 0.7), _pred("logistic", 0.56, 0.3))
    assert chosen.model_used == "logistic"
    assert chosen.direction == "DOWN"


def test_lightgbm_is_scored_when_it_wins_the_gate() -> None:
    assert selected_prediction(_pred("lightgbm", 0.58, 0.7), _pred("logistic", 0.55, 0.3)).model_used == "lightgbm"


def test_tie_goes_to_logistic_as_in_fit_ensemble() -> None:
    assert selected_prediction(_pred("lightgbm", 0.55, 0.7), _pred("logistic", 0.55, 0.3)).model_used == "logistic"


def test_missing_score_never_wins() -> None:
    assert selected_prediction(_pred("lightgbm", None, 0.7), _pred("logistic", 0.40, 0.3)).model_used == "logistic"


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_selection_matches_the_model_the_gate_was_computed_on(seed: int) -> None:
    series = synthetic_series("SYN", 1500, seed, 0.3)
    ens = fit_ensemble(build_feature_matrix("SYN", prices=series.prices))
    assert ens is not None
    chosen = selected_prediction(
        _pred("lightgbm", ens.cv_f1_lgbm, 0.6),
        _pred("logistic", ens.cv_f1_logistic, 0.4),
    )
    assert chosen.model_used == ens.model_used
