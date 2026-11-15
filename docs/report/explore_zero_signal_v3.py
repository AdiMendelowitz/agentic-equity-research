"""Exploratory diagnostic for control version 3 at zero signal. Not a pre-registered endpoint.

The zero-signal mean balanced accuracy sits above 0.50. Two explanations are separated here: a per-row dependence
between predictions and labels (a leak), or a finite-sample property of per-seed balanced accuracy when predictions
follow past returns. Pooling the confusion counts over seeds removes most of the second effect, and scoring the same
predictions against the labels of an unrelated series removes both.

Usage: python docs/report/explore_zero_signal_v3.py [--seeds N] [--start S] [--write]
Without arguments it reruns the pre-registered seeds; with --seeds it uses fresh seeds from --start, and the
intervals come from resampling whole seeds, which are independent. With --write the summary is saved as
docs/data/exploratory_zero_signal_v3.csv for the report; nothing else is written.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aer.demo import prereg_v3  # noqa: E402
from aer.demo.control import _TICKER, CONTROL_SIGMA, N_SESSIONS, _split, balanced_accuracy  # noqa: E402
from aer.demo.generate import synthetic_series  # noqa: E402
from aer.ml.constants import PREDICTION_HORIZON_DAYS  # noqa: E402
from aer.ml.features import build_feature_matrix  # noqa: E402
from aer.ml.models import fit_ensemble  # noqa: E402

PLACEBO_OFFSET: int = 5000
THRESHOLD: float = 0.5
N_RESAMPLES: int = 4000
SUMMARY_PATH = ROOT / "docs" / "data" / "exploratory_zero_signal_v3.csv"


def _matrix(seed: int) -> pd.DataFrame:
    series = synthetic_series(_TICKER, N_SESSIONS, seed, 0.0, constant_sigma=CONTROL_SIGMA)
    return build_feature_matrix(_TICKER, prices=series.prices)


def _counts(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.array(
        [
            int(np.sum((truth == 1) & (pred == 1))),
            int(np.sum(truth == 1)),
            int(np.sum((truth == 0) & (pred == 0))),
            int(np.sum(truth == 0)),
        ]
    )


def _pooled(counts: np.ndarray) -> float:
    hit_one, n_one, hit_zero, n_zero = (float(v) for v in counts)
    return (hit_one / n_one + hit_zero / n_zero) / 2


def _pooled_interval(per_seed: list[np.ndarray]) -> tuple[float, float]:
    stacked = np.vstack(per_seed)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(stacked), size=(N_RESAMPLES, len(stacked)))
    values = [_pooled(stacked[row].sum(axis=0)) for row in draws]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _describe(name: str, values: list[float]) -> str:
    arr = np.array(values)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return f"{name}: n={len(arr)} mean={arr.mean():.4f} sd={arr.std(ddof=1):.4f} se={se:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exploratory zero-signal diagnostic for control version 3.")
    parser.add_argument("--seeds", type=int, default=0, help="number of fresh seeds; 0 reruns the pre-registered seeds")
    parser.add_argument("--start", type=int, default=2000, help="first fresh seed")
    parser.add_argument("--write", action="store_true", help="save the summary under docs/data for the report")
    args = parser.parse_args(argv)
    seeds = tuple(range(args.start, args.start + args.seeds)) if args.seeds > 0 else prereg_v3.seeds()
    started = time.perf_counter()
    real: list[float] = []
    placebo: list[float] = []
    real_counts: list[np.ndarray] = []
    placebo_counts: list[np.ndarray] = []
    lag_counts = np.zeros(4, dtype=int)
    single_class = 0
    for seed in seeds:
        matrix = _matrix(seed)
        in_sample, out_sample = _split(matrix)
        fitted = fit_ensemble(in_sample)
        if fitted is None or len(out_sample) == 0:
            print(f"seed={seed} skipped")
            continue
        proba = np.asarray(fitted.predict_proba_up(out_sample[list(fitted.features)]))
        pred = (proba >= THRESHOLD).astype(int)
        truth = out_sample["label"].to_numpy().astype(int)
        lagged = matrix["label"].shift(PREDICTION_HORIZON_DAYS).loc[out_sample.index].to_numpy()
        _, other_out = _split(_matrix(seed + PLACEBO_OFFSET))
        other = other_out["label"].to_numpy().astype(int)
        n = min(len(truth), len(other))
        single_class += int(pred.min() == pred.max())
        real_ba = balanced_accuracy(truth, pred)
        placebo_ba = balanced_accuracy(other[:n], pred[:n])
        if real_ba is not None:
            real.append(real_ba)
        if placebo_ba is not None:
            placebo.append(placebo_ba)
        real_counts.append(_counts(truth, pred))
        placebo_counts.append(_counts(other[:n], pred[:n]))
        known = ~np.isnan(lagged)
        lag_counts += _counts(lagged[known].astype(int), pred[known])
        print(f"seed={seed} real_ba={real_ba} placebo_ba={placebo_ba} share_predicted_up={pred.mean():.3f}", flush=True)
    hit_one, n_one, hit_zero, n_zero = (float(v) for v in lag_counts)
    follow = hit_one / n_one - (1 - hit_zero / n_zero)
    print("EXPLORATORY, not a pre-registered endpoint")
    print(_describe("per-seed balanced accuracy, own labels", real))
    print(_describe("per-seed balanced accuracy, labels of an unrelated series", placebo))
    summary: dict[str, float] = {"seeds": float(len(real_counts)), "first_seed": float(seeds[0])}
    for key, values in (("per_seed_own", real), ("per_seed_unrelated", placebo)):
        arr = np.array(values)
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_se"] = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    for key, name, counts in (
        ("pooled_own", "own labels", real_counts),
        ("pooled_unrelated", "labels of an unrelated series", placebo_counts),
    ):
        low, high = _pooled_interval(counts)
        value = _pooled(np.vstack(counts).sum(axis=0))
        summary[key], summary[f"{key}_low"], summary[f"{key}_high"] = value, low, high
        print(f"pooled balanced accuracy, {name}: {value:.4f}, 95% seed-resampling interval ({low:.4f}, {high:.4f})")
    summary["single_class_cells"] = float(single_class)
    summary["follow_index"] = follow
    if args.write:
        lines = ["metric,value"] + [f"{key},{value:.4f}" for key, value in summary.items()]
        SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {SUMMARY_PATH.relative_to(ROOT).as_posix()}")
    print(f"cells predicting a single class: {single_class}")
    print(f"P(predict up | last completed label up) minus P(predict up | last completed label down): {follow:+.4f}")
    print("reading: pooled own-label value near 0.50 points to a finite-sample property of the per-seed statistic;")
    print("a pooled own-label value near the per-seed mean, with the unrelated-series values near 0.50, points to a")
    print("per-row dependence between predictions and labels, which would need a code review of features and labels.")
    print(f"total seconds={time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
