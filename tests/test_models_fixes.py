"""Tests for the purge gap, out-of-fold Brier score, cross-fitted calibration and FittedEnsemble."""

import dataclasses
import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from aer.ml import models as ml_models
from aer.ml.constants import (
    CV_TEST_SIZE_BARS,
    FEATURE_COLS,
    MIN_TRAINING_BARS,
    N_CV_SPLITS,
    PURGE_GAP_BARS,
)
from aer.ml.models import (
    HOLDOUT_FRACTION,
    FittedEnsemble,
    _make_lgbm,
    _make_logistic,
    _make_pipeline,
    _split_cv_holdout,
    fit_ensemble,
    train_and_predict,
)

N_LABELLED = 480
N_UNLABELLED = 3


def _frame(n_labelled: int = N_LABELLED, n_unlabelled: int = N_UNLABELLED, seed: int = 7) -> pd.DataFrame:
    """Feature frame with a nonlinear label (sign of a product), so LightGBM can beat the logistic model."""
    rng = np.random.default_rng(seed)
    n = n_labelled + n_unlabelled
    data = rng.normal(size=(n, len(FEATURE_COLS)))
    df = pd.DataFrame(data, columns=FEATURE_COLS, index=pd.bdate_range("2020-01-01", periods=n))
    signal = data[:, 0] * data[:, 1] + 0.5 * rng.normal(size=n)
    label = (signal > 0).astype(float)
    label[n_labelled:] = np.nan
    df["label"] = label
    return df


def _fixed_f1(scores):
    real = ml_models._cross_validate

    def _wrapped(pipe, X, y, splits):
        return dataclasses.replace(real(pipe, X, y, splits), f1_scores=np.array(scores))

    return _wrapped


@pytest.fixture(autouse=True)
def _few_trees(monkeypatch):
    monkeypatch.setattr(ml_models, "LGBM_N_ESTIMATORS", 30)


def _cv_block(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    labelled = df.dropna(subset=["label"])
    cv_slice, holdout_slice = _split_cv_holdout(len(labelled))
    X = labelled[[c for c in FEATURE_COLS if c in labelled.columns]]
    return X.iloc[cv_slice], labelled["label"].values.astype(int)[cv_slice], labelled.iloc[holdout_slice]


def _independent_oof(estimator, X_cv: pd.DataFrame, y_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS, gap=PURGE_GAP_BARS, test_size=CV_TEST_SIZE_BARS)
    index, probs = [], []
    for train_idx, test_idx in tscv.split(X_cv):
        pipe = _make_pipeline(estimator())
        pipe.fit(X_cv.iloc[train_idx], y_cv[train_idx])
        index.append(test_idx)
        probs.append(pipe.predict_proba(X_cv.iloc[test_idx])[:, 1])
    return np.concatenate(index), np.concatenate(probs)


@pytest.mark.parametrize("n_rows", [MIN_TRAINING_BARS, 483, 1000])
def test_split_leaves_purge_gap_and_holdout_is_the_last_fifteen_percent(n_rows):
    cv_slice, holdout_slice = _split_cv_holdout(n_rows)
    n_holdout = int(n_rows * HOLDOUT_FRACTION)
    assert HOLDOUT_FRACTION == 0.15
    assert cv_slice.start == 0
    assert holdout_slice.stop == n_rows
    assert holdout_slice.stop - holdout_slice.start == n_holdout
    assert holdout_slice.start - cv_slice.stop == PURGE_GAP_BARS


def test_no_holdout_or_gap_row_is_used_for_fitting(monkeypatch):
    df = _frame()
    labelled = df.dropna(subset=["label"])
    cv_slice, holdout_slice = _split_cv_holdout(len(labelled))
    cv_index = set(labelled.index[cv_slice])
    forbidden = set(labelled.index[cv_slice.stop :])
    assert len(forbidden) == PURGE_GAP_BARS + (holdout_slice.stop - holdout_slice.start)

    seen: list[pd.Index] = []
    original_fit = Pipeline.fit

    def _spy(self, X, y=None, **kwargs):
        seen.append(X.index)
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", _spy)
    fit_ensemble(df)

    assert len(seen) == 2 * N_CV_SPLITS + 2
    for index in seen:
        assert set(index) <= cv_index
        assert not set(index) & forbidden
    assert any(set(index) == cv_index for index in seen)


def test_calibrator_is_fitted_on_out_of_fold_probabilities(monkeypatch):
    df = _frame()
    X_cv, y_cv, _ = _cv_block(df)
    captured: dict[str, np.ndarray] = {}
    original = ml_models._fit_platt_calibrator

    def _spy(probs, y):
        captured["probs"] = np.asarray(probs)
        captured["y"] = np.asarray(y)
        return original(probs, y)

    monkeypatch.setattr(ml_models, "_fit_platt_calibrator", _spy)
    ens = fit_ensemble(df)

    assert ens is not None and ens.calibrator is not None
    assert len(captured["probs"]) == N_CV_SPLITS * CV_TEST_SIZE_BARS
    assert len(captured["y"]) == len(captured["probs"])
    oof_index, oof_probs = _independent_oof(_make_lgbm, X_cv, y_cv)
    np.testing.assert_allclose(captured["probs"], oof_probs, atol=1e-9)
    np.testing.assert_array_equal(captured["y"], y_cv[oof_index])


def test_oof_brier_equals_pooled_out_of_fold_score_and_differs_from_in_sample():
    df = _frame()
    X_cv, y_cv, _ = _cv_block(df)
    ens = fit_ensemble(df)
    assert ens is not None and ens.oof_brier is not None

    factory = _make_lgbm if ens.model_used == "lightgbm" else _make_logistic
    oof_index, oof_probs = _independent_oof(factory, X_cv, y_cv)
    assert ens.oof_brier == pytest.approx(float(brier_score_loss(y_cv[oof_index], oof_probs)), abs=1e-9)

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS, gap=PURGE_GAP_BARS, test_size=CV_TEST_SIZE_BARS)
    last_test = list(tscv.split(X_cv))[-1][1]
    best = ens.lgbm if ens.model_used == "lightgbm" else ens.logistic
    in_sample = float(brier_score_loss(y_cv[last_test], best.predict_proba(X_cv.iloc[last_test])[:, 1]))
    assert not np.isclose(ens.oof_brier, in_sample, atol=1e-4)


def test_train_and_predict_reports_out_of_fold_brier_and_holdout_check(caplog, monkeypatch):
    df = _frame()
    X_cv, y_cv, _ = _cv_block(df)
    monkeypatch.setattr(ml_models, "_cross_validate", _fixed_f1([0.6, 0.6, 0.6, 0.6, 0.6]))
    with caplog.at_level(logging.INFO, logger="aer.ml.models"):
        result = train_and_predict(df, "TEST")

    assert result is not None
    lgbm_ml, logistic_ml = result
    assert not logistic_ml.prediction_skipped
    assert logistic_ml.model_used == "logistic"
    assert lgbm_ml.brier_score_raw is None
    oof_index, oof_probs = _independent_oof(_make_logistic, X_cv, y_cv)
    assert logistic_ml.brier_score_raw == pytest.approx(float(brier_score_loss(y_cv[oof_index], oof_probs)), abs=1e-9)
    assert logistic_ml.trained_on_bars == len(X_cv)
    assert any("Brier check |" in record.message for record in caplog.records)


class _StubPipe:
    def __init__(self, probs: np.ndarray) -> None:
        self._probs = probs

    def predict_proba(self, X):
        return np.column_stack([1.0 - self._probs, self._probs])


class _StubCalibrator:
    def predict(self, probs):
        return np.clip(np.asarray(probs) * 0.5 + 0.1, 0.0, 1.0)


def _stub_ensemble(calibrator, f1_lgbm: float = 0.6, f1_logistic: float = 0.55) -> FittedEnsemble:
    return FittedEnsemble(
        features=("a", "b"),
        lgbm=_StubPipe(np.array([0.9, 0.2, 0.5])),
        logistic=_StubPipe(np.array([0.3, 0.6, 0.8])),
        calibrator=calibrator,
        cv_f1_lgbm=f1_lgbm,
        cv_f1_logistic=f1_logistic,
        cv_f1_std_lgbm=0.01,
        cv_f1_std_logistic=0.02,
        model_used="lightgbm",
        gate_passed=True,
        oof_brier=0.25,
        holdout_brier=None,
        trained_on_bars=100,
        up_rate=0.5,
    )


def test_predict_proba_up_is_the_f1_weighted_combination_with_calibrated_lightgbm():
    ens = _stub_ensemble(_StubCalibrator())
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 0.0], "c": [9.0, 9.0, 9.0]})
    weight = 0.6 / (0.6 + 0.55)
    expected = weight * (np.array([0.9, 0.2, 0.5]) * 0.5 + 0.1) + (1 - weight) * np.array([0.3, 0.6, 0.8])
    assert ens.weight_lgbm == pytest.approx(weight)
    np.testing.assert_allclose(ens.predict_proba_up(X), expected)


def test_predict_proba_up_uses_raw_lightgbm_without_calibrator_and_half_weight_on_zero_f1():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 0.0]})
    no_calibrator = _stub_ensemble(None)
    weight = 0.6 / (0.6 + 0.55)
    expected = weight * np.array([0.9, 0.2, 0.5]) + (1 - weight) * np.array([0.3, 0.6, 0.8])
    np.testing.assert_allclose(no_calibrator.predict_proba_up(X), expected)

    zero = _stub_ensemble(None, f1_lgbm=0.0, f1_logistic=0.0)
    assert zero.weight_lgbm == 0.5
    equal_mix = 0.5 * np.array([0.9, 0.2, 0.5]) + 0.5 * np.array([0.3, 0.6, 0.8])
    np.testing.assert_allclose(zero.predict_proba_up(X), equal_mix)


def test_predict_proba_up_of_a_real_fit_matches_hand_computation():
    df = _frame()
    ens = fit_ensemble(df)
    assert ens is not None
    X = df[list(ens.features)].iloc[-30:]
    raw = ens.lgbm.predict_proba(X)[:, 1]
    lgbm_prob = ens.calibrator.predict(raw) if ens.calibrator is not None else raw
    logistic_prob = ens.logistic.predict_proba(X)[:, 1]
    weight = ens.cv_f1_lgbm / (ens.cv_f1_lgbm + ens.cv_f1_logistic)
    np.testing.assert_allclose(ens.predict_proba_up(X), weight * lgbm_prob + (1 - weight) * logistic_prob)
    assert ens.holdout_brier is not None
    assert set(ens.holdout_brier) == {"lgbm", "logistic", "ensemble"}
    assert ens.trained_on_bars == len(_cv_block(df)[0])


def test_fit_ensemble_returns_none_below_minimum_training_bars():
    df = _frame(n_labelled=MIN_TRAINING_BARS - 1)
    assert fit_ensemble(df) is None


def test_fit_ensemble_still_fits_both_models_when_the_gate_fails(monkeypatch):
    df = _frame()
    monkeypatch.setattr(ml_models, "_cross_validate", _fixed_f1([0.3, 0.7, 0.2, 0.8, 0.1]))
    ens = fit_ensemble(df)
    assert ens is not None
    assert ens.gate_passed is False
    X = df[list(ens.features)].iloc[-5:]
    assert ens.lgbm.predict_proba(X).shape == (5, 2)
    assert ens.logistic.predict_proba(X).shape == (5, 2)
    result = train_and_predict(df, "TEST")
    assert result is not None
    assert result[0].prediction_skipped is True
    assert result[0].model_used == "none"
