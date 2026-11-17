"""Regenerate the report figures from the aggregates in docs/data as deterministic SVG files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import NoReturn

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.stats import t as student_t
from sklearn.model_selection import TimeSeriesSplit

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "svg.hashsalt": "build_figures-v1",
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "docs" / "data"
DEFAULT_OUT = ROOT / "docs" / "img"
sys.path.insert(0, str(ROOT / "src"))

from aer.demo.prereg_v3 import ALPHA, DELTA, SIGNALS  # noqa: E402
from aer.ml.constants import CV_TEST_SIZE_BARS, MIN_TRAINING_BARS, N_CV_SPLITS, PURGE_GAP_BARS  # noqa: E402
from aer.ml.models import _split_cv_holdout  # noqa: E402

# The first five colours are the report's; purple and green match the README diagrams.
PALETTE = {
    "blue": "#4C78A8",
    "orange": "#C08A4A",
    "vermillion": "#B5604F",
    "grey": "#8A94A3",
    "dark": "#374151",
    "purple": "#7A6AA6",
    "green": "#4E8F6E",
}
MIN_N = 10
Z95 = 1.959963984540054
FOOTER = "Whole-sample aggregates; no ticker, week or date dimension is published."

HEADERS: dict[str, tuple[str, ...]] = {
    "eval_calibration.csv": ("conviction_bin", "n_resolved", "n_correct"),
    "eval_recommendation.csv": ("recommendation", "n_rows", "n_resolved", "n_synth_correct"),
    "eval_ml_gate.csv": ("cv_f1_bin", "n_rows", "n_skipped", "n_ml_resolved", "n_ml_correct"),
    "eval_backtest_sharpe.csv": ("sharpe_bin", "n_rows"),
    "run_history.csv": ("run_date", "outcome", "exit_code"),
    "detection_curve.csv": (
        "signal",
        "seed",
        "gate_predicted",
        "gate_reason",
        "cv_f1_mean",
        "cv_f1_std",
        "model_balanced_accuracy",
        "oracle_balanced_accuracy",
        "majority_accuracy",
        "label_base_rate",
        "n_oos",
    ),
    "detection_curve_v3.csv": (
        "signal",
        "seed",
        "gate_passed",
        "model_used",
        "cv_f1_mean",
        "cv_f1_std",
        "model_balanced_accuracy",
        "oracle_balanced_accuracy",
        "holdout_brier_ensemble",
        "majority_accuracy",
        "label_base_rate",
        "n_oos",
    ),
}
LLM_KEYS = ("provider", "model", "calls", "in_tokens", "out_tokens")


def fail(message: str) -> NoReturn:
    raise SystemExit(f"build_figures: {message}")


def read_rows(name: str) -> list[dict[str, str]]:
    path = DATA_DIR / name
    if not path.is_file():
        fail(f"missing data file {path.relative_to(ROOT).as_posix()}")
    reader = csv.DictReader(path.read_text(encoding="utf-8").splitlines())
    header = tuple(reader.fieldnames or ())
    if header != HEADERS[name]:
        fail(f"unexpected header in {name}: expected {','.join(HEADERS[name])}, found {','.join(header)}")
    return list(reader)


def read_llm_models() -> tuple[int, list[dict[str, object]]]:
    path = DATA_DIR / "llm_usage.jsonl"
    if not path.is_file():
        fail(f"missing data file {path.relative_to(ROOT).as_posix()}")
    runs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    models: list[dict[str, object]] = []
    for run in runs:
        for entry in run.get("by_model", []):
            missing = [key for key in LLM_KEYS if key not in entry]
            if missing:
                fail(f"llm_usage.jsonl entry lacks keys {','.join(missing)}")
            models.append(entry)
    return len(runs), models


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bin_midpoint(label: str) -> float | None:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[-:]\s*(\d+(?:\.\d+)?)\s*", label)
    return (float(match.group(1)) + float(match.group(2))) / 2 if match else None


def bin_key(label: str) -> float:
    mid = bin_midpoint(label)
    return mid if mid is not None else float(label)


def ordered_bins(labels: list[str]) -> list[str]:
    try:
        return sorted(labels, key=bin_key)
    except ValueError:
        return list(labels)


def sorted_by_bin(rows: list[dict[str, str]], column: str) -> list[dict[str, str]]:
    lookup = {r[column]: r for r in rows}
    return [lookup[label] for label in ordered_bins([r[column] for r in rows])]


def sharpe_rows() -> list[dict[str, str]]:
    return sorted_by_bin(read_rows("eval_backtest_sharpe.csv"), "sharpe_bin")


def sharpe_bin_order() -> list[str]:
    return [r["sharpe_bin"] for r in sharpe_rows()]


def add_footer(fig: Figure, extra: str | None = None) -> None:
    fig.text(0.01, 0.012, FOOTER, fontsize=9, color=PALETTE["dark"])
    if extra:
        fig.text(0.01, 0.04, extra, fontsize=9, color=PALETTE["dark"])


def add_legend(fig: Figure, handles: list[Artist], ncol: int = 2, y: float = 0.045) -> None:
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=9, frameon=False)


def rate_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", color=PALETTE["blue"], linestyle="none", label=f"n >= {MIN_N}"),
        Line2D(
            [],
            [],
            marker="o",
            color=PALETTE["grey"],
            markerfacecolor="none",
            linestyle="none",
            label=f"n < {MIN_N} (hollow, grey: interpret with caution)",
        ),
    ]


def count_handle() -> Patch:
    return Patch(
        facecolor="none", edgecolor=PALETTE["grey"], linewidth=1.4, label=f"fewer than {MIN_N} rows in the bin"
    )


def draw_rates(ax: Axes, xs: list[float], ks: list[int], ns: list[int]) -> None:
    for x, k, n in zip(xs, ks, ns, strict=True):
        if n == 0:
            continue
        lo, hi = wilson(k, n)
        p = k / n
        colour = PALETTE["grey"] if n < MIN_N else PALETTE["blue"]
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=colour, elinewidth=1.4, capsize=4)
        ax.scatter(
            [x],
            [p],
            s=60,
            facecolors="none" if n < MIN_N else colour,
            edgecolors=colour,
            linewidths=1.6,
            zorder=3,
        )
        ax.annotate(f"n={n}", (x, hi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)


def reference_line(ax: Axes) -> Line2D:
    (line,) = ax.plot([], [], color=PALETTE["dark"], linestyle="--", linewidth=1, label="0.5 reference")
    ax.axhline(0.5, color=PALETTE["dark"], linestyle="--", linewidth=1)
    return line


def fig_calibration() -> Figure:
    rows = sorted_by_bin(read_rows("eval_calibration.csv"), "conviction_bin")
    labels = [r["conviction_bin"] for r in rows]
    ns = [int(r["n_resolved"]) for r in rows]
    ks = [int(r["n_correct"]) for r in rows]
    mids = [bin_midpoint(label) for label in labels]
    numeric = all(m is not None for m in mids)
    xs = [float(m) for m in mids if m is not None] if numeric else [float(i) for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    handles = [reference_line(ax)]
    if numeric:
        (ident,) = ax.plot([0, 1], [0, 1], color=PALETTE["vermillion"], linewidth=1.2, label="perfect calibration")
        handles.append(ident)
    else:
        ax.set_xticks(xs, labels)
    draw_rates(ax, xs, ks, ns)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Conviction bin (as labelled in the data)")
    ax.set_ylabel("Hit rate (resolved calls)")
    ax.set_title(f"Hit rate by conviction bin with 95% Wilson intervals (n={sum(ns)} resolved)", fontsize=11)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    add_legend(fig, handles + rate_handles())
    add_footer(fig)
    return fig


def fig_recommendation() -> Figure:
    rows = sorted(read_rows("eval_recommendation.csv"), key=lambda r: r["recommendation"])
    labels = [r["recommendation"] for r in rows]
    ns = [int(r["n_resolved"]) for r in rows]
    ks = [int(r["n_synth_correct"]) for r in rows]
    xs = [float(i) for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    handles = [reference_line(ax)]
    ax.set_xticks(xs, labels)
    draw_rates(ax, xs, ks, ns)
    for x, n_rows, n in zip(xs, (int(r["n_rows"]) for r in rows), ns, strict=True):
        if n == 0:
            ax.text(x, 0.05, f"no resolved rows\n(n_rows={n_rows})", ha="center", fontsize=9, color=PALETTE["grey"])
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("Recommendation")
    ax.set_ylabel("Hit rate (resolved rows)")
    ax.set_title(f"Hit rate by recommendation with 95% Wilson intervals (n={sum(ns)} resolved)", fontsize=11)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    add_legend(fig, handles + rate_handles())
    add_footer(fig)
    return fig


def fig_ml_gate() -> Figure:
    rows = sorted_by_bin(read_rows("eval_ml_gate.csv"), "cv_f1_bin")
    labels = [r["cv_f1_bin"] for r in rows]
    n_rows = [int(r["n_rows"]) for r in rows]
    skipped = [int(r["n_skipped"]) for r in rows]
    n_ml = [int(r["n_ml_resolved"]) for r in rows]
    k_ml = [int(r["n_ml_correct"]) for r in rows]
    xs = [float(i) for i in range(len(labels))]
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 6))
    for x, s, n in zip(xs, skipped, n_rows, strict=True):
        small = n < MIN_N
        colour = PALETTE["grey"] if small else PALETTE["orange"]
        top.bar(x, s / n, width=0.6, facecolor="none" if small else colour, edgecolor=colour, linewidth=1.4)
        top.annotate(f"n={n}", (x, s / n), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    top.set_ylim(0, 1.12)
    top.set_ylabel("Share abstained")
    top.set_title("(a) Rows on which the gate abstained", loc="left", fontsize=10)
    handles = [reference_line(bottom)]
    draw_rates(bottom, xs, k_ml, n_ml)
    bottom.set_xticks(xs, labels)
    bottom.set_ylim(0, 1.12)
    bottom.set_xlabel("Cross-validated F1 bin (as labelled in the data)")
    bottom.set_ylabel("Accuracy")
    bottom.set_title("(b) Accuracy where the gate predicted, 95% Wilson intervals", loc="left", fontsize=10)
    n_predicted = sum(n - s for n, s in zip(n_rows, skipped, strict=True))
    fig.suptitle(
        f"ML gate by cross-validated F1 bin ({sum(n_rows):,} rows, {n_predicted} predictions, {sum(n_ml)} resolved)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 0.97))
    add_legend(fig, handles + rate_handles() + [count_handle()])
    add_footer(fig)
    return fig


def fig_sharpe() -> Figure:
    rows = sharpe_rows()
    labels = [r["sharpe_bin"] for r in rows]
    ns = [int(r["n_rows"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, n in enumerate(ns):
        small = n < MIN_N
        colour = PALETTE["grey"] if small else PALETTE["blue"]
        ax.bar(i, n, width=0.6, facecolor="none" if small else colour, edgecolor=colour, linewidth=1.4)
        ax.annotate(f"n={n}", (i, n), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, max(ns) * 1.12)
    ax.set_xlabel("Sharpe-ratio bin (as labelled in the data)")
    ax.set_ylabel("Rows (count)")
    ax.set_title(f"Rows per backtest Sharpe-ratio bin (n={sum(ns)} rows)", fontsize=11)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    add_legend(fig, [count_handle()], ncol=1, y=0.075)
    add_footer(fig, "Counts of rows per bin only; no trading result is implied.")
    return fig


def fig_runs() -> Figure:
    rows = sorted(read_rows("run_history.csv"), key=lambda r: r["run_date"])
    dates = [date.fromisoformat(r["run_date"]) for r in rows]
    states = [
        0 if r["outcome"] == "complete" and r["exit_code"].strip() == "0" else 1 if r["outcome"] == "complete" else 2
        for r in rows
    ]
    spec = (
        (2, "complete, exit code 0", PALETTE["blue"], "o"),
        # square, not diamond: a rotated marker path prints -0 or 0 depending on the platform maths library
        (1, "complete, non-zero exit code", PALETTE["orange"], "s"),
        (0, "no end line logged", PALETTE["vermillion"], "X"),
    )
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for y, name, colour, marker in spec:
        state = 2 - y
        xs = [d for d, st in zip(dates, states, strict=True) if st == state]
        ax.scatter(xs, [y] * len(xs), s=60, color=colour, marker=marker, label=f"{name} (n={len(xs)})")
    ax.set_yticks([2, 1, 0], [name for _, name, _, _ in spec])
    ax.set_ylim(-0.6, 2.6)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.set_xlabel("Scheduled run date (day)")
    ax.set_ylabel("Run outcome")
    n_ok = states.count(0)
    ax.set_title(
        f"Scheduled runs: {n_ok} of {len(rows)} ({100 * n_ok / len(rows):.0f}%) completed with exit code 0", fontsize=11
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    handles, _ = ax.get_legend_handles_labels()
    add_legend(fig, list(handles), ncol=3, y=0.01)
    return fig


def fig_llm() -> Figure:
    n_runs, entries = read_llm_models()
    totals: dict[str, list[int]] = {}
    for e in entries:
        key = f"{e['provider']}: {e['model']}"
        acc = totals.setdefault(key, [0, 0, 0])
        acc[0] += int(str(e["calls"]))
        acc[1] += int(str(e["in_tokens"]))
        acc[2] += int(str(e["out_tokens"]))
    names = sorted(totals, key=lambda k: (-totals[k][0], k))
    ys = list(range(len(names)))[::-1]
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(9.4, 3.9))
    left.barh(ys, [totals[k][0] for k in names], color=PALETTE["blue"])
    right.barh(ys, [totals[k][1] for k in names], color=PALETTE["orange"], label="input tokens")
    right.barh(
        ys,
        [totals[k][2] for k in names],
        left=[totals[k][1] for k in names],
        color=PALETTE["vermillion"],
        label="output tokens",
    )
    for y, k in zip(ys, names, strict=True):
        left.annotate(
            f"{totals[k][0]:,}", (totals[k][0], y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9
        )
        end = totals[k][1] + totals[k][2]
        right.annotate(f"{end:,}", (end, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    left.set_yticks(ys, names)
    thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")
    millions = FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M")
    for ax, label, fmt in ((left, "Model calls (count)", thousands), (right, "Tokens (millions)", millions)):
        ax.xaxis.set_major_formatter(fmt)
        ax.set_xlabel(label)
        ax.margins(x=0.3)
    left.set_ylabel("Provider and model")
    left.set_title("(a) Calls", loc="left", fontsize=10)
    right.set_title("(b) Tokens", loc="left", fontsize=10)
    fig.suptitle(f"LLM usage by provider and model, summed over {n_runs} logged runs", fontsize=11)
    fig.tight_layout(rect=(0, 0.14, 1, 0.97))
    handles, _ = right.get_legend_handles_labels()
    add_legend(fig, list(handles), ncol=2, y=0.045)
    fig.text(0.01, 0.012, "Grouping: provider and model; counts as logged per run.", fontsize=9, color=PALETTE["dark"])
    return fig


def fig_detection() -> Figure:
    rows = read_rows("detection_curve.csv")
    signals = sorted({float(r["signal"]) for r in rows})
    seeds = len({r["seed"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # A constant majority-class prediction has balanced accuracy 0.50, so the 0.5 line is that baseline; the raw
    # majority accuracy in the data is a different metric and is not drawn on this axis.
    reference = reference_line(ax)
    reference.set_label("0.50, chance and majority-class balanced accuracy")
    handles = [reference]
    oracle_means: list[float] = []
    for x in signals:
        group = [r for r in rows if float(r["signal"]) == x]
        accs = [float(r["model_balanced_accuracy"]) for r in group if r["model_balanced_accuracy"]]
        oracle = [float(r["oracle_balanced_accuracy"]) for r in group if r["oracle_balanced_accuracy"]]
        predicted = sum(1 for r in group if r["gate_predicted"] == "True")
        mean = sum(accs) / len(accs) if accs else math.nan
        sd = math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) if len(accs) > 1 else 0.0
        half = Z95 * sd / math.sqrt(len(accs)) if accs else 0.0
        oracle_means.append(sum(oracle) / len(oracle) if oracle else math.nan)
        ax.errorbar(x, mean, yerr=half, fmt="o", color=PALETTE["blue"], capsize=4, elinewidth=1.4, zorder=3)
        ax.annotate(
            f"gate predicted {predicted}/{len(group)}",
            (x, 0.39),
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["dark"],
        )
    rho2 = [18 * s * s / (5 + 21 * s * s + 2 * math.sqrt(10) * s) for s in signals]
    ceiling = [0.5 + math.asin(math.sqrt(v)) / math.pi for v in rho2]
    (oracle_line,) = ax.plot(signals, oracle_means, color=PALETTE["vermillion"], linewidth=1.4, label="oracle")
    (ceiling_line,) = ax.plot(
        signals, ceiling, color=PALETTE["orange"], linewidth=1.4, linestyle=":", label="analytic ceiling"
    )
    ax.set_ylim(0.38, 0.85)
    ax.set_xlim(signals[0] - 0.09, signals[-1] + 0.09)
    ax.set_xlabel("Injected signal strength")
    ax.set_ylabel("Balanced accuracy (out of sample)")
    ax.set_title(
        f"Control v2 signal detection, 95% normal interval ({seeds} seeds, {len(rows)} cells)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    mean_handle = Line2D(
        [], [], marker="o", color=PALETTE["blue"], linestyle="none", label="model, 95% normal interval (1.96)"
    )
    add_legend(fig, handles + [oracle_line, ceiling_line, mean_handle], ncol=2)
    note = "Synthetic data with a known injected signal; no market data."
    fig.text(0.01, 0.012, note, fontsize=9, color=PALETTE["dark"])
    return fig


def rule_baselines() -> tuple[int, int, int, int]:
    """Correct pipeline calls, correct always-down calls, correct always-up calls and resolved calls.

    The scorer counts a BUY as correct when the five-day move is up and a SELL or AVOID as correct when it is down,
    so the down moves are the incorrect BUY calls plus the correct SELL and AVOID calls.
    """
    rows = {r["recommendation"]: r for r in read_rows("eval_recommendation.csv")}
    if set(rows) != {"BUY", "SELL", "AVOID", "HOLD"}:
        fail(f"unexpected recommendations in eval_recommendation.csv: {','.join(sorted(rows))}")
    if int(rows["HOLD"]["n_resolved"]):
        fail("HOLD rows are resolved; the rule baselines assume none are")
    resolved = sum(int(r["n_resolved"]) for r in rows.values())
    correct = sum(int(r["n_synth_correct"]) for r in rows.values())
    buy = rows["BUY"]
    down = int(buy["n_resolved"]) - int(buy["n_synth_correct"])
    down += sum(int(rows[k]["n_synth_correct"]) for k in ("SELL", "AVOID"))
    return correct, down, resolved - down, resolved


@dataclass(frozen=True)
class V3Point:
    signal: float
    values: tuple[float, ...]
    mean: float
    lower: float
    upper: float
    oracle: float
    gate_passed: int
    seeds: int


def v3_points() -> list[V3Point]:
    """Per-signal summary of control version 3, with the pre-registered t interval.

    The interval is mean +/- t(1 - ALPHA, n - 1) * sd / sqrt(n): at zero signal it is the equivalence interval, and
    at the other signals its lower end is the one-sided bound of the superiority test.
    """
    rows = read_rows("detection_curve_v3.csv")
    points: list[V3Point] = []
    for signal in SIGNALS:
        group = [r for r in rows if math.isclose(float(r["signal"]), signal)]
        values = tuple(float(r["model_balanced_accuracy"]) for r in group if r["model_balanced_accuracy"])
        oracle = [float(r["oracle_balanced_accuracy"]) for r in group if r["oracle_balanced_accuracy"]]
        if len(values) < 2 or not oracle:
            fail(f"detection_curve_v3.csv holds fewer than two scored seeds at signal {signal}")
        mean = statistics.fmean(values)
        half = float(student_t.ppf(1 - ALPHA, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
        points.append(
            V3Point(
                signal=signal,
                values=values,
                mean=mean,
                lower=mean - half,
                upper=mean + half,
                oracle=statistics.fmean(oracle),
                gate_passed=sum(1 for r in group if r["gate_passed"] == "True"),
                seeds=len(group),
            )
        )
    return points


def draw_v3(ax: Axes, points: list[V3Point], compact: bool) -> list[Artist]:
    band = ax.axhspan(0.5 - DELTA, 0.5 + DELTA, color=PALETTE["grey"], alpha=0.18, linewidth=0)
    band.set_label(f"pre-registered margin, {0.5 - DELTA:.2f} to {0.5 + DELTA:.2f}")
    reference = reference_line(ax)
    reference.set_label("0.50, chance balanced accuracy")
    xs = [p.signal for p in points]
    for p in points:
        # Deterministic horizontal spread for the per-seed points; no random jitter, so the build stays byte-identical.
        spread = [p.signal + ((i % 11) - 5) * 0.012 for i in range(len(p.values))]
        ax.scatter(spread, p.values, s=9 if compact else 12, color=PALETTE["purple"], alpha=0.35, linewidths=0)
        ax.errorbar(
            p.signal,
            p.mean,
            yerr=[[p.mean - p.lower], [p.upper - p.mean]],
            fmt="o",
            color=PALETTE["blue"],
            markersize=6,
            capsize=4,
            elinewidth=1.6,
            zorder=4,
        )
        if not compact:
            ax.annotate(f"{p.mean:.3f}", (p.signal, p.upper), xytext=(14, 2), textcoords="offset points", fontsize=9)
            ax.annotate(
                f"gate passed {p.gate_passed}/{p.seeds}",
                (p.signal, 0.385),
                ha="center",
                va="bottom",
                fontsize=9,
                color=PALETTE["dark"],
            )
    (oracle_line,) = ax.plot(xs, [p.oracle for p in points], color=PALETTE["vermillion"], linewidth=1.4)
    oracle_line.set_label("oracle on the true conditional mean")
    seeds = Line2D([], [], marker="o", color=PALETTE["purple"], alpha=0.5, linestyle="none", label="one seed")
    mean = Line2D(
        [],
        [],
        marker="o",
        color=PALETTE["blue"],
        linestyle="none",
        label=f"model mean, {100 * (1 - 2 * ALPHA):.0f}% t interval",
    )
    ax.set_xticks(xs, [f"{x:.1f}" for x in xs])
    ax.set_xlim(xs[0] - 0.12, xs[-1] + 0.12)
    ax.set_ylim(0.38, 0.8)
    return [mean, seeds, oracle_line, band, reference]


def fig_detection_v3() -> Figure:
    points = v3_points()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    handles = draw_v3(ax, points, compact=False)
    ax.set_xlabel("Injected signal strength")
    ax.set_ylabel("Balanced accuracy (out of sample)")
    ax.set_title(
        f"Control v3, pre-registered, {points[0].seeds} seeds per strength (ensemble, every seed scored)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    add_legend(fig, handles, ncol=2, y=0.05)
    note = "Synthetic data with a known injected signal; no market data."
    fig.text(0.01, 0.012, note, fontsize=9, color=PALETTE["dark"])
    return fig


def cv_layout(n_rows: int) -> tuple[list[tuple[range, range]], range, range]:
    """Training and validation rows of each fold, the CV block and the holdout block, from the code's own splitter."""
    cv_slice, holdout_slice = _split_cv_holdout(n_rows)
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS, gap=PURGE_GAP_BARS, test_size=CV_TEST_SIZE_BARS)
    folds = [
        (range(int(train[0]), int(train[-1]) + 1), range(int(test[0]), int(test[-1]) + 1))
        for train, test in tscv.split(np.arange(cv_slice.stop))
    ]
    return folds, range(0, cv_slice.stop), range(holdout_slice.start, n_rows)


def fig_cv_geometry() -> Figure:
    n_rows = MIN_TRAINING_BARS
    folds, cv_block, holdout = cv_layout(n_rows)
    colours = {"train": PALETTE["blue"], "gap": "#D9DCE1", "valid": PALETTE["orange"], "holdout": PALETTE["green"]}
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    labels: list[tuple[int, str]] = []
    for i, (train, valid) in enumerate(folds):
        y = len(folds) - i
        ax.broken_barh([(train.start, len(train))], (y - 0.32, 0.64), color=colours["train"])
        ax.broken_barh([(train.stop, valid.start - train.stop)], (y - 0.32, 0.64), color=colours["gap"])
        ax.broken_barh([(valid.start, len(valid))], (y - 0.32, 0.64), color=colours["valid"])
        labels.append((y, f"fold {i + 1}"))
    ax.broken_barh([(cv_block.start, len(cv_block))], (-0.32, 0.64), color=colours["train"], alpha=0.55)
    ax.broken_barh([(cv_block.stop, holdout.start - cv_block.stop)], (-0.32, 0.64), color=colours["gap"])
    ax.broken_barh([(holdout.start, len(holdout))], (-0.32, 0.64), color=colours["holdout"])
    labels.append((0, "final fit"))
    ax.set_yticks([y for y, _ in labels], [name for _, name in labels])
    ax.set_xlim(0, n_rows)
    ax.set_ylim(-0.8, len(folds) + 0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Labelled row, in time order")
    ax.set_title(
        f"Purged time-series cross-validation at the minimum of {n_rows} labelled rows, drawn from the code",
        fontsize=10,
    )
    handles = [
        Patch(color=colours["train"], label="training rows"),
        Patch(color=colours["gap"], label=f"purge gap, {PURGE_GAP_BARS} rows dropped"),
        Patch(color=colours["valid"], label=f"validation fold, {CV_TEST_SIZE_BARS} rows"),
        Patch(color=colours["holdout"], label=f"holdout block, {len(holdout)} rows, scored only"),
    ]
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    add_legend(fig, list(handles), ncol=2, y=0.03)
    return fig


def fig_hero() -> Figure:
    correct, down, up, resolved = rule_baselines()
    n_runs, entries = read_llm_models()
    paid = sum(int(str(e["calls"])) for e in entries if e["provider"] == "anthropic")
    free = sum(int(str(e["calls"])) for e in entries if e["provider"] != "anthropic")
    widths = {"width_ratios": (1.15, 1.3, 0.9)}
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(12.4, 4.2), gridspec_kw=widths)

    rules = [
        ("pipeline calls", correct, PALETTE["blue"]),
        ("always down", down, PALETTE["grey"]),
        ("always up", up, PALETTE["grey"]),
    ]
    for y, (name, k, colour) in zip((2, 1, 0), rules, strict=True):
        p = k / resolved
        lo, hi = wilson(k, resolved)
        left.errorbar(p, y, xerr=[[p - lo], [hi - p]], fmt="o", color=colour, capsize=4)
        left.annotate(f"{k}/{resolved}", (hi, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    left.axvline(0.5, color=PALETTE["dark"], linestyle="--", linewidth=1)
    left.set_yticks([2, 1, 0], [name for name, _, _ in rules])
    left.set_xlim(0.38, 0.68)
    left.set_ylim(-0.6, 2.6)
    left.set_xlabel("Share correct, 95% Wilson interval")
    left.set_title("(a) Recorded calls, one per ticker-day", loc="left", fontsize=10)

    v3_handles = draw_v3(middle, v3_points(), compact=True)
    middle.legend(handles=[v3_handles[0], v3_handles[3]], loc="upper left", fontsize=8, frameon=False)
    middle.set_xlabel("Injected signal strength")
    middle.set_ylabel("Balanced accuracy")
    middle.set_title("(b) Pre-registered control, known signal", loc="left", fontsize=10)

    total = free + paid
    right.barh([1, 0], [free, paid], color=[PALETTE["green"], PALETTE["orange"]], height=0.55)
    for y, n in ((1, free), (0, paid)):
        right.annotate(
            f"{n:,} ({100 * n / total:.0f}%)",
            (n, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    right.set_yticks([1, 0], ["free providers", "paid fallback"])
    right.set_xlim(0, free * 1.45)
    right.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    right.set_xlabel("Model calls (count)")
    right.set_title(f"(c) Routing, {n_runs} logged runs", loc="left", fontsize=10)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    note = (
        "Calls scored on the direction of the five-day move. "
        "The control uses synthetic prices with an injected signal."
    )
    fig.text(0.01, 0.012, note, fontsize=9, color=PALETTE["dark"])
    return fig


FIGURES: tuple[tuple[str, Callable[[], Figure]], ...] = (
    ("calibration.svg", fig_calibration),
    ("recommendation_hit_rate.svg", fig_recommendation),
    ("ml_gate.svg", fig_ml_gate),
    ("backtest_sharpe_distribution.svg", fig_sharpe),
    ("run_reliability.svg", fig_runs),
    ("llm_usage.svg", fig_llm),
    ("detection_curve_v3.svg", fig_detection_v3),
    ("cv_geometry.svg", fig_cv_geometry),
    ("hero_evidence.svg", fig_hero),
) + ((("detection_curve.svg", fig_detection),) if (DATA_DIR / "detection_curve.csv").is_file() else ())


def save(fig: Figure, path: Path) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": "build_figures"})
    plt.close(fig)
    data = buf.getvalue().replace(b"\r\n", b"\n")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate report figures from docs/data.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory (default: docs/img)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not (DATA_DIR / "detection_curve.csv").is_file():
        print("detection_curve.svg skipped: docs/data/detection_curve.csv is absent (run docs/report/run_control.py)")
    for name, build in FIGURES:
        print(f"{name} {save(build(), args.out / name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
