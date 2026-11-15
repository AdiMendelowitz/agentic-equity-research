"""Run the pre-registered detection control (version 3) and write docs/data/detection_curve_v3.csv.

Usage: python docs/report/run_control_v3.py [--one-cell]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aer.demo import prereg_v3  # noqa: E402
from aer.demo.control_v3 import CellV3, analyse, run_cell_v3, write_csv_v3  # noqa: E402

CSV_PATH = ROOT / "docs" / "data" / "detection_curve_v3.csv"


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _p(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _verdict(value: bool | None) -> str:
    return "NOT EVALUATED" if value is None else "MET" if value else "NOT MET"


def describe(cell: CellV3, seconds: float) -> str:
    return (
        f"signal={cell.signal} seed={cell.seed} gate_passed={cell.gate_passed} model_used={cell.model_used} "
        f"cv_f1_mean={_f(cell.cv_f1_mean)} model_ba={_f(cell.model_balanced_accuracy)} "
        f"oracle_ba={_f(cell.oracle_balanced_accuracy)} brier={_f(cell.holdout_brier_ensemble)} "
        f"majority={cell.majority_accuracy:.4f} base_rate={cell.label_base_rate:.4f} n_oos={cell.n_oos} "
        f"seconds={seconds:.1f}"
    )


def format_result(result: dict[str, Any]) -> list[str]:
    eq = result["equivalence_zero_signal"]
    lines = [
        f"spec={result['spec_id']} delta={result['delta']} alpha={result['alpha']}",
        "primary (a), zero signal, equivalence to chance by two one-sided tests:",
        f"  n={eq['n']} mean={_f(eq['mean'])} sd={_f(eq['sd'])} "
        f"{eq['interval_confidence']:.0%} interval=({_f(eq['lower'])}, {_f(eq['upper'])}) "
        f"margin=({eq['margin_low']:.4f}, {eq['margin_high']:.4f}) verdict={_verdict(eq['verdict'])}",
        "primary (b), non-zero signals, superiority over 0.50 (one-sided t test, Holm-adjusted):",
    ]
    for signal, row in result["superiority"].items():
        lines.append(
            f"  signal={signal} n={row['n']} mean={_f(row['mean'])} sd={_f(row['sd'])} "
            f"p_raw={_p(row['p_raw'])} p_holm={_p(row['p_holm'])} lower_bound={_f(row['lower_bound'])} "
            f"verdict={_verdict(row['verdict'])}"
        )
    lines.append("secondary (c), paired gap oracle minus model:")
    for signal, row in result["paired_gap"].items():
        lines.append(
            f"  signal={signal} n={row['n']} mean={_f(row['mean'])} "
            f"{row['interval_confidence']:.0%} interval=({_f(row['lower'])}, {_f(row['upper'])}) "
            f"analytic_ceiling={row['analytic_ceiling']:.4f}"
        )
    lines.append("descriptive (d), gate:")
    for signal, row in result["gate"].items():
        lines.append(
            f"  signal={signal} gate_passed={row['gate_passed']} of {row['n_cells']} cells ({row['n_fitted']} fitted)"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pre-registered detection control, version 3.")
    parser.add_argument(
        "--one-cell",
        action="store_true",
        help="run only signal 0.6 and the first seed and print its timing; allowed before N_SEEDS is set",
    )
    args = parser.parse_args(argv)
    if args.one_cell:
        pairs = [(0.6, prereg_v3.SEED_OFFSET)]
    else:
        if prereg_v3.N_SEEDS is None:
            print(
                "refusing to run: prereg_v3.N_SEEDS is None. Run docs/report/power_v3.py, set N_SEEDS to its "
                "output and publish the specification hash before the full run.",
                file=sys.stderr,
            )
            return 2
        pairs = [(signal, seed) for signal in prereg_v3.SIGNALS for seed in prereg_v3.seeds()]
    cells: list[CellV3] = []
    started = time.perf_counter()
    for signal, seed in pairs:
        start = time.perf_counter()
        cell = run_cell_v3(signal, seed)
        cells.append(cell)
        print(describe(cell, time.perf_counter() - start), flush=True)
    if args.one_cell:
        print(f"total seconds={time.perf_counter() - started:.1f}")
        return 0
    write_csv_v3(cells, CSV_PATH)
    print(f"wrote {CSV_PATH.relative_to(ROOT).as_posix()}")
    for line in format_result(analyse(cells)):
        print(line)
    print(f"total seconds={time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
