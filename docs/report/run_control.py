"""Run the pre-registered detection control (version 2) and write docs/data/detection_curve.csv.

Usage: python docs/report/run_control.py [--one-cell]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aer.demo.control import (  # noqa: E402
    SEEDS,
    SIGNALS,
    ControlCell,
    evaluate_expectations,
    run_cell,
    summarise,
    write_csv,
)

CSV_PATH = ROOT / "docs" / "data" / "detection_curve.csv"
MODEL_DIR = ROOT / "artifacts" / "control"


def describe(cell: ControlCell, seconds: float) -> str:
    return (
        f"signal={cell.signal} seed={cell.seed} gate={cell.gate_reason} cv_f1_mean={cell.cv_f1_mean} "
        f"model_ba={cell.model_balanced_accuracy} oracle_ba={cell.oracle_balanced_accuracy} "
        f"majority={cell.majority_accuracy:.4f} base_rate={cell.label_base_rate:.4f} n_oos={cell.n_oos} "
        f"seconds={seconds:.1f}"
    )


def verdict(value: bool | None) -> str:
    return "NOT EVALUATED" if value is None else "MET" if value else "NOT MET"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pre-registered detection control.")
    parser.add_argument("--one-cell", action="store_true", help="run only signal 0.6, seed 0 and print its timing")
    args = parser.parse_args(argv)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [(0.6, 0)] if args.one_cell else [(signal, seed) for signal in SIGNALS for seed in SEEDS]
    cells: list[ControlCell] = []
    started = time.perf_counter()
    for signal, seed in pairs:
        start = time.perf_counter()
        cell = run_cell(signal, seed, MODEL_DIR)
        cells.append(cell)
        print(describe(cell, time.perf_counter() - start), flush=True)
    if args.one_cell:
        print(f"total seconds={time.perf_counter() - started:.1f}")
        return 0
    write_csv(cells, CSV_PATH)
    print(f"wrote {CSV_PATH.relative_to(ROOT).as_posix()}")
    for row in summarise(cells):
        print(row)
    for name, value in evaluate_expectations(cells).items():
        print(f"{name}: {verdict(value)}")
    print(f"total seconds={time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
