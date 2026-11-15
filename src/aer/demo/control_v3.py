"""Pre-registered detection control, version 3: scores the deployed calibrated ensemble against a matched oracle.

The rules in analyse() implement docs/prereg/control-v3.md exactly. No endpoint other than those is computed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

from aer.demo.control import (
    _TICKER,
    CONTROL_SIGMA,
    N_SESSIONS,
    _fmt,
    _oracle_balanced_accuracy,
    _split,
    analytic_ceiling,
    balanced_accuracy,
)
from aer.demo.generate import synthetic_series
from aer.demo.prereg_v3 import ALPHA, DELTA, SPEC_ID
from aer.ml.features import build_feature_matrix
from aer.ml.models import fit_ensemble

CSV_HEADER_V3: str = (
    "signal,seed,gate_passed,model_used,cv_f1_mean,cv_f1_std,model_balanced_accuracy,"
    "oracle_balanced_accuracy,holdout_brier_ensemble,majority_accuracy,label_base_rate,n_oos"
)
_CHANCE = 0.5
_DECISION_THRESHOLD = 0.5


@dataclass(frozen=True)
class CellV3:
    signal: float
    seed: int
    gate_passed: bool
    model_used: str | None
    cv_f1_mean: float | None
    cv_f1_std: float | None
    model_balanced_accuracy: float | None
    oracle_balanced_accuracy: float | None
    holdout_brier_ensemble: float | None
    majority_accuracy: float
    label_base_rate: float
    n_oos: int


def run_cell_v3(signal: float, seed: int) -> CellV3:
    series = synthetic_series(_TICKER, N_SESSIONS, seed, signal, constant_sigma=CONTROL_SIGMA)
    matrix = build_feature_matrix(_TICKER, prices=series.prices)
    in_sample, out_sample = _split(matrix)
    base_rate = float(out_sample["label"].mean()) if len(out_sample) else 0.5
    oracle = _oracle_balanced_accuracy(series.expected_forward, out_sample) if len(out_sample) else None
    fitted = fit_ensemble(in_sample)
    cv_mean: float | None = None
    cv_std: float | None = None
    model_ba: float | None = None
    brier: float | None = None
    model_used: str | None = None
    gate_passed = False
    if fitted is not None:
        model_used = fitted.model_used
        gate_passed = bool(fitted.gate_passed)
        lightgbm = fitted.model_used == "lightgbm"
        cv_mean = fitted.cv_f1_lgbm if lightgbm else fitted.cv_f1_logistic
        cv_std = fitted.cv_f1_std_lgbm if lightgbm else fitted.cv_f1_std_logistic
        if fitted.holdout_brier is not None:
            brier = fitted.holdout_brier.get("ensemble")
        if len(out_sample):
            proba = fitted.predict_proba_up(out_sample[list(fitted.features)])
            predicted = (np.asarray(proba) >= _DECISION_THRESHOLD).astype(int)
            model_ba = balanced_accuracy(out_sample["label"].to_numpy(), predicted)
    return CellV3(
        signal=signal,
        seed=seed,
        gate_passed=gate_passed,
        model_used=model_used,
        cv_f1_mean=cv_mean,
        cv_f1_std=cv_std,
        model_balanced_accuracy=model_ba,
        oracle_balanced_accuracy=oracle,
        holdout_brier_ensemble=brier,
        majority_accuracy=max(base_rate, 1 - base_rate),
        label_base_rate=base_rate,
        n_oos=len(out_sample),
    )


def write_csv_v3(cells: Sequence[CellV3], path: Path) -> None:
    lines = [CSV_HEADER_V3]
    for c in cells:
        fields = [
            _fmt(c.signal),
            str(c.seed),
            str(c.gate_passed),
            "" if c.model_used is None else c.model_used,
            _fmt(c.cv_f1_mean),
            _fmt(c.cv_f1_std),
            _fmt(c.model_balanced_accuracy),
            _fmt(c.oracle_balanced_accuracy),
            _fmt(c.holdout_brier_ensemble),
            _fmt(c.majority_accuracy),
            _fmt(c.label_base_rate),
            str(c.n_oos),
        ]
        lines.append(",".join(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment: the i-th smallest p (0-based rank i) is scaled by (m - i), made monotone, capped at 1."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _signals(cells: Sequence[CellV3]) -> list[float]:
    return sorted({c.signal for c in cells})


def _values(cells: Sequence[CellV3], signal: float, attr: str) -> list[float]:
    found = (getattr(c, attr) for c in cells if c.signal == signal)
    return [float(v) for v in found if v is not None]


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def _equivalence(values: Sequence[float], delta: float, alpha: float) -> dict[str, Any]:
    n = len(values)
    result: dict[str, Any] = {
        "n": n,
        "mean": None,
        "sd": None,
        "interval_confidence": 1 - 2 * alpha,
        "lower": None,
        "upper": None,
        "margin_low": _CHANCE - delta,
        "margin_high": _CHANCE + delta,
        "verdict": None,
    }
    if n < 2:
        return result
    mean, sd = _mean_sd(values)
    half = float(t.ppf(1 - alpha, n - 1)) * sd / math.sqrt(n)
    lower, upper = mean - half, mean + half
    result.update(
        mean=mean,
        sd=sd,
        lower=lower,
        upper=upper,
        verdict=bool(_CHANCE - delta < lower and upper < _CHANCE + delta),
    )
    return result


def _one_sided_p(mean: float, sd: float, n: int) -> tuple[float, float]:
    if sd == 0.0:
        if mean > _CHANCE:
            return math.inf, 0.0
        return (-math.inf, 1.0) if mean < _CHANCE else (0.0, 0.5)
    statistic = (mean - _CHANCE) / (sd / math.sqrt(n))
    return statistic, float(t.sf(statistic, n - 1))


def _superiority(per_signal: dict[float, list[float]], alpha: float) -> dict[float, dict[str, Any]]:
    rows: dict[float, dict[str, Any]] = {}
    raw: dict[float, float] = {}
    for signal, values in per_signal.items():
        n = len(values)
        row: dict[str, Any] = {
            "n": n,
            "mean": None,
            "sd": None,
            "t_statistic": None,
            "p_raw": None,
            "p_holm": None,
            "lower_bound": None,
            "verdict": None,
        }
        if n >= 2:
            mean, sd = _mean_sd(values)
            statistic, p_value = _one_sided_p(mean, sd, n)
            row.update(
                mean=mean,
                sd=sd,
                t_statistic=statistic,
                p_raw=p_value,
                lower_bound=mean - float(t.ppf(1 - alpha, n - 1)) * sd / math.sqrt(n),
            )
            raw[signal] = p_value
        rows[signal] = row
    # m stays the number of non-zero signals; a signal without enough data cannot be claimed and counts as p = 1
    ordered = list(per_signal)
    adjusted = holm_adjust([raw.get(signal, 1.0) for signal in ordered])
    for signal, p_holm in zip(ordered, adjusted, strict=True):
        if signal in raw:
            rows[signal]["p_holm"] = p_holm
            rows[signal]["verdict"] = bool(p_holm < alpha)
    return rows


def _paired_gap(cells: Sequence[CellV3], signal: float, alpha: float) -> dict[str, Any]:
    gaps = [
        c.oracle_balanced_accuracy - c.model_balanced_accuracy
        for c in cells
        if c.signal == signal and c.oracle_balanced_accuracy is not None and c.model_balanced_accuracy is not None
    ]
    row: dict[str, Any] = {
        "n": len(gaps),
        "mean": None,
        "lower": None,
        "upper": None,
        "interval_confidence": 1 - alpha,
        "analytic_ceiling": analytic_ceiling(signal),
    }
    if len(gaps) >= 2:
        mean, sd = _mean_sd(gaps)
        half = float(t.ppf(1 - alpha / 2, len(gaps) - 1)) * sd / math.sqrt(len(gaps))
        row.update(mean=mean, lower=mean - half, upper=mean + half)
    return row


def analyse(cells: Sequence[CellV3], delta: float = DELTA, alpha: float = ALPHA) -> dict[str, Any]:
    signals = _signals(cells)
    nonzero = [s for s in signals if s != 0.0]
    zero_values = _values(cells, 0.0, "model_balanced_accuracy")
    group = {s: [c for c in cells if c.signal == s] for s in signals}
    return {
        "spec_id": SPEC_ID,
        "delta": delta,
        "alpha": alpha,
        "equivalence_zero_signal": _equivalence(zero_values, delta, alpha),
        "superiority": _superiority({s: _values(cells, s, "model_balanced_accuracy") for s in nonzero}, alpha),
        "paired_gap": {s: _paired_gap(cells, s, alpha) for s in signals},
        "gate": {
            s: {
                "n_cells": len(group[s]),
                "n_fitted": sum(1 for c in group[s] if c.model_used is not None),
                "gate_passed": sum(1 for c in group[s] if c.gate_passed),
            }
            for s in signals
        },
    }
