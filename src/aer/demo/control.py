"""Pre-registered detection control (version 2): does the gate, and a fitted model, find a signal of known strength?

The signal is a moving sum of past shocks with constant volatility, so the best possible balanced accuracy has a
closed form (analytic_ceiling). The expectations V2-E0..V2-E4 were fixed before any run and are evaluated from the
cells without tuning.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import aer.ml.models as ml_models
from aer.demo.generate import synthetic_series
from aer.ml.constants import (
    CV_F1_MEAN_MINIMUM,
    CV_F1_STD_HARD_LIMIT,
    FEATURE_COLS,
    PURGE_GAP_BARS,
    ROUND_TRIP_COST_BPS,
)
from aer.ml.features import build_feature_matrix

SIGNALS: tuple[float, ...] = (0.0, 0.3, 0.6)
SEEDS: tuple[int, ...] = tuple(range(20))
N_SESSIONS: int = 1500
OOS_FRACTION: float = 0.2
CONTROL_SIGMA: float = 0.015
CSV_HEADER: str = (
    "signal,seed,gate_predicted,gate_reason,cv_f1_mean,cv_f1_std,model_balanced_accuracy,"
    "oracle_balanced_accuracy,majority_accuracy,label_base_rate,n_oos"
)
_TICKER = "SYN1"
_Z95 = 1.96
_CEILING_TOLERANCE = 0.05
GATE_REASONS: tuple[str, ...] = (
    "predicted",
    "too_few_bars",
    "f1_mean_below_minimum",
    "f1_std_above_limit",
    "f1_mean_below_minimum_and_std_above_limit",
    "none_returned",
)


@dataclass(frozen=True)
class ControlCell:
    signal: float
    seed: int
    gate_predicted: bool
    gate_reason: str
    cv_f1_mean: float | None
    cv_f1_std: float | None
    model_balanced_accuracy: float | None
    oracle_balanced_accuracy: float | None
    majority_accuracy: float
    label_base_rate: float
    n_oos: int


def balanced_accuracy(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> float | None:
    truth = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    positives = truth == 1
    negatives = truth == 0
    if not positives.any() or not negatives.any():
        return None
    recall_one = float(np.mean(pred[positives] == 1))
    recall_zero = float(np.mean(pred[negatives] == 0))
    return (recall_one + recall_zero) / 2


def analytic_ceiling(signal: float) -> float:
    rho2 = 18 * signal**2 / (5 + 21 * signal**2 + 2 * math.sqrt(10) * signal)
    return 0.5 + math.asin(math.sqrt(rho2)) / math.pi


def _split(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_oos = int(len(matrix) * OOS_FRACTION)
    cut = len(matrix) - n_oos
    in_sample = matrix.iloc[: cut - PURGE_GAP_BARS].dropna(subset=["label"])
    out_sample = matrix.iloc[cut:].dropna(subset=["label"])
    return in_sample, out_sample


def _skip_reason(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "too_few_bars"
    low = mean < CV_F1_MEAN_MINIMUM
    high = std is not None and std > CV_F1_STD_HARD_LIMIT
    if low and high:
        return "f1_mean_below_minimum_and_std_above_limit"
    if low:
        return "f1_mean_below_minimum"
    if high:
        return "f1_std_above_limit"
    return "none_returned"


@dataclass(frozen=True)
class _GateOutcome:
    predicted: bool
    reason: str
    cv_f1_mean: float | None
    cv_f1_std: float | None
    family: str


def _gate(in_sample: pd.DataFrame, model_dir: Path) -> _GateOutcome:
    original = ml_models.MODEL_DIR
    ml_models.MODEL_DIR = model_dir
    try:
        result = ml_models.train_and_predict(in_sample, _TICKER)
    finally:
        ml_models.MODEL_DIR = original
    if result is None:
        return _GateOutcome(False, "none_returned", None, None, "logistic")
    lgbm_ml, logistic_ml = result
    if lgbm_ml.prediction_skipped:
        mean, std = lgbm_ml.cv_f1_macro, lgbm_ml.cv_f1_std
        return _GateOutcome(False, _skip_reason(mean, std), mean, std, "logistic")
    lgbm_f1 = lgbm_ml.cv_f1_macro if lgbm_ml.cv_f1_macro is not None else 0.0
    logistic_f1 = logistic_ml.cv_f1_macro if logistic_ml.cv_f1_macro is not None else 0.0
    winner = lgbm_ml if lgbm_f1 > logistic_f1 else logistic_ml
    family = "lightgbm" if lgbm_f1 > logistic_f1 else "logistic"
    return _GateOutcome(True, "predicted", winner.cv_f1_macro, winner.cv_f1_std, family)


def _model_balanced_accuracy(in_sample: pd.DataFrame, out_sample: pd.DataFrame, family: str) -> float | None:
    features = [c for c in FEATURE_COLS if c in in_sample.columns]
    estimator = ml_models._make_lgbm() if family == "lightgbm" else ml_models._make_logistic()
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)]).set_output(transform="pandas")
    pipe.fit(in_sample[features], in_sample["label"].values.astype(int))
    return balanced_accuracy(out_sample["label"].to_numpy(), pipe.predict(out_sample[features]))


def _oracle_balanced_accuracy(expected_forward: pd.Series, out_sample: pd.DataFrame) -> float | None:
    threshold = math.log(1 + ROUND_TRIP_COST_BPS / 10_000)
    predicted = (expected_forward.loc[out_sample.index].to_numpy() > threshold).astype(int)
    return balanced_accuracy(out_sample["label"].to_numpy(), predicted)


def run_cell(signal: float, seed: int, model_dir: Path) -> ControlCell:
    series = synthetic_series(_TICKER, N_SESSIONS, seed, signal, constant_sigma=CONTROL_SIGMA)
    matrix = build_feature_matrix(_TICKER, prices=series.prices)
    in_sample, out_sample = _split(matrix)
    gate = _gate(in_sample, model_dir)
    base_rate = float(out_sample["label"].mean()) if len(out_sample) else 0.5
    model_ba = _model_balanced_accuracy(in_sample, out_sample, gate.family) if len(out_sample) else None
    return ControlCell(
        signal=signal,
        seed=seed,
        gate_predicted=gate.predicted,
        gate_reason=gate.reason,
        cv_f1_mean=gate.cv_f1_mean,
        cv_f1_std=gate.cv_f1_std,
        model_balanced_accuracy=model_ba,
        oracle_balanced_accuracy=_oracle_balanced_accuracy(series.expected_forward, out_sample),
        majority_accuracy=max(base_rate, 1 - base_rate),
        label_base_rate=base_rate,
        n_oos=len(out_sample),
    )


def run_control(
    model_dir: Path,
    signals: Sequence[float] = SIGNALS,
    seeds: Sequence[int] = SEEDS,
) -> list[ControlCell]:
    return [run_cell(signal, seed, model_dir) for signal in signals for seed in seeds]


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_csv(cells: Sequence[ControlCell], path: Path) -> None:
    lines = [CSV_HEADER]
    for c in cells:
        fields = [
            _fmt(c.signal),
            str(c.seed),
            str(c.gate_predicted),
            c.gate_reason,
            _fmt(c.cv_f1_mean),
            _fmt(c.cv_f1_std),
            _fmt(c.model_balanced_accuracy),
            _fmt(c.oracle_balanced_accuracy),
            _fmt(c.majority_accuracy),
            _fmt(c.label_base_rate),
            str(c.n_oos),
        ]
        lines.append(",".join(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def _values(cells: Sequence[ControlCell], signal: float, attr: str) -> list[float]:
    found = (getattr(c, attr) for c in cells if c.signal == signal)
    return [float(v) for v in found if v is not None]


def _stats(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    mean = float(np.mean(values))
    return mean, _Z95 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def _model_stats(cells: Sequence[ControlCell], signal: float) -> tuple[float, float] | None:
    return _stats(_values(cells, signal, "model_balanced_accuracy"))


def _oracle_stats(cells: Sequence[ControlCell], signal: float) -> tuple[float, float] | None:
    return _stats(_values(cells, signal, "oracle_balanced_accuracy"))


def _signals(cells: Sequence[ControlCell]) -> list[float]:
    return sorted({c.signal for c in cells})


def _e0(cells: Sequence[ControlCell]) -> bool | None:
    verdicts: list[bool] = []
    for signal in _signals(cells):
        stats = _oracle_stats(cells, signal)
        if stats is None:
            return None
        if signal == 0.0:
            verdicts.append(abs(stats[0] - 0.5) <= 1e-9)
        else:
            verdicts.append(abs(stats[0] - analytic_ceiling(signal)) <= _CEILING_TOLERANCE)
    return all(verdicts) if verdicts else None


def _e1(cells: Sequence[ControlCell]) -> bool | None:
    stats = _model_stats(cells, 0.0)
    return None if stats is None else abs(stats[0] - 0.5) <= stats[1]


def _e2(cells: Sequence[ControlCell]) -> bool | None:
    stats = _model_stats(cells, 0.6)
    return None if stats is None else stats[0] - stats[1] > 0.5


def _e3(cells: Sequence[ControlCell]) -> bool | None:
    means: list[float] = []
    for signal in _signals(cells):
        stats = _model_stats(cells, signal)
        if stats is None:
            return None
        means.append(stats[0])
    return all(a <= b for a, b in zip(means, means[1:], strict=False)) if means else None


def _e4(cells: Sequence[ControlCell]) -> bool | None:
    verdicts: list[bool] = []
    for signal in _signals(cells):
        model = _model_stats(cells, signal)
        oracle = _oracle_stats(cells, signal)
        if model is None or oracle is None:
            return None
        verdicts.append(model[0] <= oracle[0] + model[1])
    return all(verdicts) if verdicts else None


def evaluate_expectations(cells: Sequence[ControlCell]) -> dict[str, bool | None]:
    return {
        "V2-E0": _e0(cells),
        "V2-E1": _e1(cells),
        "V2-E2": _e2(cells),
        "V2-E3": _e3(cells),
        "V2-E4": _e4(cells),
    }


def summarise(cells: Sequence[ControlCell]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for signal in _signals(cells):
        group = [c for c in cells if c.signal == signal]
        model = _model_stats(cells, signal)
        oracle = _oracle_stats(cells, signal)
        majority = _values(cells, signal, "majority_accuracy")
        rows.append(
            {
                "signal": signal,
                "n": len(group),
                "model_ba_mean": None if model is None else model[0],
                "model_ba_half_width": None if model is None else model[1],
                "oracle_ba_mean": None if oracle is None else oracle[0],
                "analytic_ceiling": analytic_ceiling(signal),
                "majority_accuracy_mean": float(np.mean(majority)) if majority else None,
                "gate_predicted_share": sum(1 for c in group if c.gate_predicted) / len(group),
                "gate_reasons": {r: sum(1 for c in group if c.gate_reason == r) for r in GATE_REASONS},
            }
        )
    return rows
