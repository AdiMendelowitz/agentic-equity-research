"""Sample-size calculation for the pre-registered detection control, version 3.

Reads the version 2 pilot (docs/data/detection_curve.csv) for standard deviations only, inflates each to its upper
one-sided confidence limit, and prints the seed count the specification needs.

Usage: python docs/report/power_v3.py
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from scipy.stats import chi2, nct, t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aer.demo import prereg_v3  # noqa: E402

PILOT_PATH = ROOT / "docs" / "data" / "detection_curve.csv"

PowerFn = Callable[[int, float, float, float], float]


def sd_upper(sd: float, n: int, confidence: float) -> float:
    return sd * math.sqrt((n - 1) / float(chi2.ppf(1 - confidence, n - 1)))


def power_superiority(n: int, delta: float, sd: float, alpha: float) -> float:
    critical = float(t.ppf(1 - alpha, n - 1))
    return 1.0 - float(nct.cdf(critical, n - 1, delta * math.sqrt(n) / sd))


def power_tost(n: int, delta: float, sd: float, alpha: float) -> float:
    """Conservative lower bound on the power of the two one-sided tests at a true mean of 0.50.

    Each one-sided test fails to reject with probability p; the bound 1 - 2p ignores the overlap of the two
    failures, so the exact joint power is at least this value.
    """
    critical = float(t.ppf(1 - alpha, n - 1))
    return max(0.0, 1.0 - 2.0 * float(nct.cdf(critical, n - 1, delta * math.sqrt(n) / sd)))


def required_seeds(
    power_fn: PowerFn,
    delta: float,
    sd: float,
    alpha: float,
    target: float,
    minimum: int,
    maximum: int = 500,
) -> int:
    for n in range(max(minimum, 2), maximum + 1):
        if power_fn(n, delta, sd, alpha) >= target:
            return n
    raise ValueError(f"no n up to {maximum} reaches power {target} for delta={delta}, sd={sd}, alpha={alpha}")


def main(argv: list[str] | None = None) -> int:
    pilot = pd.read_csv(PILOT_PATH, encoding="utf-8")
    rows: list[tuple[float, int, float, float]] = []
    for signal in prereg_v3.SIGNALS:
        values = pilot.loc[(pilot["signal"] - signal).abs() < 1e-9, "model_balanced_accuracy"].dropna()
        if len(values) < 2:
            print(f"pilot has fewer than two values for signal {signal}", file=sys.stderr)
            return 2
        sd = float(values.std(ddof=1))
        rows.append((signal, len(values), sd, sd_upper(sd, len(values), prereg_v3.SD_UPPER_CONFIDENCE)))
    print(f"pilot={PILOT_PATH.relative_to(ROOT).as_posix()} delta={prereg_v3.DELTA} alpha={prereg_v3.ALPHA}")
    print(f"power={prereg_v3.POWER} sd_upper_confidence={prereg_v3.SD_UPPER_CONFIDENCE}")
    print("signal  n_pilot  sd_pilot  sd_upper")
    for signal, n, sd, upper in rows:
        print(f"{signal:<6}  {n:<7}  {sd:.5f}   {upper:.5f}")
    zero_sd = next(upper for signal, _, _, upper in rows if signal == 0.0)
    nonzero_sds = [upper for signal, _, _, upper in rows if signal != 0.0]
    m = len(nonzero_sds)
    n_equivalence = required_seeds(
        power_tost, prereg_v3.DELTA, zero_sd, prereg_v3.ALPHA, prereg_v3.POWER, prereg_v3.MIN_SEEDS
    )
    n_superiority = required_seeds(
        power_superiority, prereg_v3.DELTA, max(nonzero_sds), prereg_v3.ALPHA / m, prereg_v3.POWER, prereg_v3.MIN_SEEDS
    )
    print(f"n_equivalence (zero signal, sd {zero_sd:.5f}, alpha {prereg_v3.ALPHA}) = {n_equivalence}")
    print(f"n_superiority (largest non-zero sd {max(nonzero_sds):.5f}, alpha {prereg_v3.ALPHA}/{m}) = {n_superiority}")
    print(f"minimum = {prereg_v3.MIN_SEEDS}")
    print(f"N_SEEDS = {max(n_equivalence, n_superiority, prereg_v3.MIN_SEEDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
