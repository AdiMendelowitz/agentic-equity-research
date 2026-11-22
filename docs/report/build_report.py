"""Build the static evaluation report (one HTML page plus figures) from the committed data.

Every count, rate and interval on the page is computed here from a file under docs/data, except the historical
correction table (CORRECTION_ROWS), whose counts are literals. The design effects are assumed values set in this
module, and the footer records the hash of every file in docs/data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import os
import platform
import shutil
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aer.demo.control_v3 import CellV3, analyse  # noqa: E402
from aer.demo.prereg_v3 import SIGNALS as PREREG_SIGNALS  # noqa: E402

DATA_DIR = ROOT / "docs" / "data"
IMG_DIR = ROOT / "docs" / "img"
SPEC_FILE = ROOT / "src" / "aer" / "evaluation" / "spec.py"
DEFAULT_OUT = ROOT / "site"
CONFIDENCE = 0.95
CHANCE = 0.5
GATE_VALUES = {"True": True, "False": False}

FIGURES: tuple[tuple[str, str], ...] = (
    (
        "detection_curve.svg",
        "Control version 2, model and oracle balanced accuracy against signal strength, with ceiling and 0.50 line",
    ),
    (
        "detection_curve_v3.svg",
        "Control version 3, pre-registered, 46 seeds per signal strength, per-seed and mean balanced accuracy with "
        "t intervals against signal strength, with the oracle line, the pre-registered margin band and the 0.50 line",
    ),
    ("calibration.svg", "Hit rate by conviction bin with 95% Wilson intervals, against the 0.50 line"),
    ("recommendation_hit_rate.svg", "Hit rate by recommendation with 95% Wilson intervals"),
    (
        "ml_gate.svg",
        "Share of rows on which the ML gate abstained, and the accuracy of the recorded LightGBM direction "
        "on the predictions that resolved",
    ),
)

CSS = """
body{font-family:Arial,Helvetica,sans-serif;color:#1F2937;background:#FFFFFF;max-width:860px;margin:0 auto;
padding:24px;line-height:1.5}
h1{font-size:1.8rem;margin-bottom:.25rem}h2{font-size:1.3rem;margin-top:2rem;border-bottom:1px solid #C9CED6}
h3{font-size:1.05rem;margin-top:1.5rem}
table{border-collapse:collapse;margin:1rem 0}th,td{padding:.35rem .75rem;border-bottom:1px solid #E5E7EB;
text-align:left}td.n{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:1.5rem 0}figure img{max-width:100%;height:auto}figcaption,.source{font-size:.9rem;color:#4B5563}
footer{margin-top:3rem;font-size:.85rem;color:#4B5563}code{font-size:.85rem}
a{color:#2F5F8F}a:focus,a:hover{outline:2px solid #2F5F8F;outline-offset:2px}
"""


@dataclass(frozen=True)
class CohortStep:
    label: str
    count: int


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True)
class SeedRow:
    seed: int
    gate_predicted: bool
    model: float | None
    oracle: float | None
    n_oos: int

    @property
    def gap(self) -> float | None:
        return None if self.model is None or self.oracle is None else self.oracle - self.model


@dataclass(frozen=True)
class SignalSummary:
    signal: str
    seeds: tuple[SeedRow, ...]
    model: Interval | None
    oracle: Interval | None
    gap: Interval | None

    @property
    def gate_predicted(self) -> int:
        return sum(1 for row in self.seeds if row.gate_predicted)


@dataclass(frozen=True)
class GateComparison:
    first: SignalSummary
    last: SignalSummary
    p_value: float
    paired_p_value: float
    discordant: tuple[int, int]


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def t_critical(n: int) -> float:
    return float(stats.t.ppf(0.5 + CONFIDENCE / 2, n - 1))


def t_interval(values: Sequence[float]) -> Interval | None:
    n = len(values)
    if n < 2:
        return None
    centre = statistics.fmean(values)
    half = t_critical(n) * statistics.stdev(values) / math.sqrt(n)
    return Interval(centre, centre - half, centre + half, n)


def cohort() -> list[CohortStep]:
    rows = read_csv("eval_recommendation.csv")
    briefs = sum(int(r["n_rows"]) for r in rows)
    hold = sum(int(r["n_rows"]) for r in rows if r["recommendation"] == "HOLD")
    resolved = sum(int(r["n_resolved"]) for r in rows if r["recommendation"] != "HOLD")
    correct = sum(int(r["n_synth_correct"]) for r in rows if r["recommendation"] != "HOLD")
    return [
        CohortStep("Ticker-days in the published evaluation data", briefs),
        CohortStep("HOLD, no scored outcome", hold),
        CohortStep("Actionable calls (BUY, SELL or AVOID)", briefs - hold),
        CohortStep("Actionable calls not resolved", briefs - hold - resolved),
        CohortStep("Resolved against a five-day return", resolved),
        CohortStep("Correct", correct),
    ]


def _optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def _seed_row(row: dict[str, str]) -> SeedRow:
    return SeedRow(
        seed=int(row["seed"]),
        gate_predicted=GATE_VALUES[row["gate_predicted"]],
        model=_optional_float(row["model_balanced_accuracy"]),
        oracle=_optional_float(row["oracle_balanced_accuracy"]),
        n_oos=int(row["n_oos"]),
    )


def control() -> list[SignalSummary]:
    rows = read_csv("detection_curve.csv")
    out: list[SignalSummary] = []
    for signal in sorted({r["signal"] for r in rows}, key=float):
        seeds = tuple(sorted((_seed_row(r) for r in rows if r["signal"] == signal), key=lambda s: s.seed))
        out.append(
            SignalSummary(
                signal=f"{float(signal):.2f}",
                seeds=seeds,
                model=t_interval([s.model for s in seeds if s.model is not None]),
                oracle=t_interval([s.oracle for s in seeds if s.oracle is not None]),
                gap=t_interval([g for s in seeds if (g := s.gap) is not None]),
            )
        )
    return out


def gate_comparison(summaries: Sequence[SignalSummary]) -> GateComparison | None:
    if len(summaries) < 2:
        return None
    first, last = summaries[0], summaries[-1]
    table = [
        [first.gate_predicted, len(first.seeds) - first.gate_predicted],
        [last.gate_predicted, len(last.seeds) - last.gate_predicted],
    ]
    _, p_value = stats.fisher_exact(table, alternative="two-sided")
    # The same seed ids run at every signal, so the outcomes are paired; the exact test on the discordant seeds
    # (McNemar) is the design-matched comparison. Fisher's test is kept as first reported.
    last_by_seed = {s.seed: s.gate_predicted for s in last.seeds}
    if {s.seed for s in first.seeds} != set(last_by_seed) or len(last_by_seed) != len(last.seeds):
        raise ValueError("gate comparison needs the same seed ids at the first and last signal")
    pairs = [(s.gate_predicted, last_by_seed[s.seed]) for s in first.seeds]
    only_first = sum(1 for a, b in pairs if a and not b)
    only_last = sum(1 for a, b in pairs if b and not a)
    n_discordant = only_first + only_last
    paired_p = 1.0 if n_discordant == 0 else float(stats.binomtest(only_first, n_discordant, 0.5).pvalue)
    return GateComparison(first, last, float(p_value), paired_p, (only_first, only_last))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_interval(interval: Interval | None) -> str:
    if interval is None:
        return "n/a"
    return f"{interval.mean:.3f} ({interval.low:.3f} to {interval.high:.3f})"


def contains_chance(interval: Interval | None) -> str:
    if interval is None:
        return "n/a"
    return "yes" if interval.contains(CHANCE) else "no"


def _gate_label(row: SeedRow) -> str:
    return "predicted" if row.gate_predicted else "abstained"


# Sensitivity of the 17 November 2026 correction, as printed by the private extractor from the 9 November snapshot:
# (rule, correct calls, resolved calls, correct ML predictions, resolved ML predictions). Historical, so fixed here.
CORRECTION_ROWS: tuple[tuple[str, int, int, int, int], ...] = (
    ("Every row, as first published on 9 November", 342, 707, 74, 145),
    ("First run of each ticker-day (used on this page)", 314, 629, 64, 120),
    ("Last run of each ticker-day", 314, 629, 63, 121),
)


def correction_note() -> str:
    body = "".join(
        f'<tr><td>{html.escape(rule)}</td><td class="n">{k:,} of {n:,} ({k / n:.3f})</td>'
        f'<td class="n">{mk} of {mn} ({mk / mn:.3f})</td></tr>'
        for rule, k, n, mk, mn in CORRECTION_ROWS
    )
    return (
        "<p><strong>Correction, 17 November 2026.</strong> The aggregates first published on 9 November counted "
        "same-day reruns of the pipeline as separate calls. Of the 2,609 rows first published, 288 were second or "
        "later runs of a ticker-day already in the store. The 78 of them that were resolved calls had been "
        "scored against the same five-day outcome as the first run, so they were not independent observations. "
        "Every figure on recorded output on this page now counts one decision per ticker-day and keeps the first "
        "run of each day. "
        "Under each counting rule in the table below, recorded accuracy stays close to 0.50.</p>"
        '<table id="correction"><caption>Scored cohort under each counting rule</caption><thead><tr>'
        '<th scope="col">Rule</th><th scope="col">Recorded calls correct</th>'
        '<th scope="col">ML predictions correct</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def cohort_section() -> str:
    steps = cohort()
    resolved, correct = steps[4].count, steps[5].count
    body = "".join(f'<tr><td>{html.escape(s.label)}</td><td class="n">{s.count:,}</td></tr>' for s in steps)
    return (
        '<section id="cohort"><h2>Evaluation cohort</h2>'
        "<p>Accuracy is measured on resolved actionable calls only, one decision per ticker-day. Each step below "
        "is computed from the data file named under the table.</p>"
        f'<table><caption>From all ticker-days to the scored cohort</caption><thead><tr><th scope="col">Step</th>'
        f'<th scope="col">Rows</th></tr></thead><tbody>{body}</tbody></table>'
        f"<p>Accuracy on the scored cohort is {correct:,} of {resolved:,} ({correct / resolved:.3f}).</p>"
        '<p class="source">Source: <code>docs/data/eval_recommendation.csv</code></p>'
        f"{correction_note()}</section>"
    )


def _summary_row(s: SignalSummary) -> str:
    oracle_mean = None if s.oracle is None else s.oracle.mean
    n_model = 0 if s.model is None else s.model.n
    return (
        f'<tr><td class="n">{s.signal}</td><td class="n">{n_model}</td>'
        f'<td class="n">{_fmt_interval(s.model)}</td><td>{contains_chance(s.model)}</td>'
        f'<td class="n">{_fmt(oracle_mean)}</td><td class="n">{_fmt_interval(s.gap)}</td>'
        f'<td class="n">{s.gate_predicted} of {len(s.seeds)}</td></tr>'
    )


def _t_note(summaries: Sequence[SignalSummary]) -> str:
    sizes = sorted({s.model.n for s in summaries if s.model is not None})
    if len(sizes) == 1:
        return f"t = {t_critical(sizes[0]):.3f} with {sizes[0] - 1} degrees of freedom"
    return "t with n - 1 degrees of freedom, n being the seed count in each row"


def _gate_note(summaries: Sequence[SignalSummary]) -> str:
    comparison = gate_comparison(summaries)
    if comparison is None:
        return ""
    first, last = comparison.first, comparison.last
    return (
        f"<p>The gate predicted on {first.gate_predicted} of {len(first.seeds)} seeds at signal {first.signal} "
        f"and on {last.gate_predicted} of {len(last.seeds)} at signal {last.signal}. The same seeds run at both "
        f"strengths, so the comparison is paired, with {comparison.discordant[0]} seeds predicted only at "
        f"{first.signal} and {comparison.discordant[1]} only at {last.signal} (exact McNemar test, two-sided "
        f"p = {comparison.paired_p_value:.3f}, added after the results were seen; Fisher's exact test, which "
        f"treats the seeds as independent and was reported first, gives p = {comparison.p_value:.3f}). "
        "The paired comparison found no clear change in how often the gate passed.</p>"
    )


def detection_section(summaries: Sequence[SignalSummary]) -> str:
    rows = "".join(_summary_row(s) for s in summaries)
    return (
        '<section id="control"><h2>Control experiment, version 2</h2>'
        "<p>This is version 2 of the control, kept on record beside the pre-registered version 3 in the next section. "
        "Synthetic prices with a known injected signal; no market data. Seeds are treated as independent "
        "replications. Every interval in the table below is a 95% Student's t interval over seeds "
        f"({_t_note(summaries)}). The oracle-minus-model gap is computed within each seed, so it is paired.</p>"
        "<p>The expectations fixed before the run, in <code>src/aer/demo/control.py</code>, use a normal "
        "multiplier of 1.96, and the README table and the detection figure report those narrower intervals. "
        "The table below reports t intervals; the column on chance is computed from them.</p>"
        '<table id="control-summary"><caption>Out-of-sample balanced accuracy by injected signal</caption>'
        '<thead><tr><th scope="col">Signal</th><th scope="col">Seeds</th>'
        '<th scope="col">Model, mean (95% t interval)</th><th scope="col">Interval contains 0.50</th>'
        '<th scope="col">Oracle, mean</th><th scope="col">Oracle minus model, mean (95% t interval)</th>'
        f'<th scope="col">Gate predicted</th></tr></thead><tbody>{rows}</tbody></table>'
        f"{_gate_note(summaries)}"
        '<p class="source">Source: <code>docs/data/detection_curve.csv</code></p></section>'
    )


def _seed_table(summary: SignalSummary) -> str:
    body = "".join(
        f'<tr><td class="n">{r.seed}</td><td>{_gate_label(r)}</td><td class="n">{_fmt(r.model)}</td>'
        f'<td class="n">{_fmt(r.oracle)}</td><td class="n">{_fmt(r.gap)}</td><td class="n">{r.n_oos}</td></tr>'
        for r in summary.seeds
    )
    return (
        f"<h3>Signal {summary.signal}</h3>"
        f'<table id="seeds-{summary.signal}"><caption>Per-seed results at signal {summary.signal}, '
        "version 2</caption>"
        '<thead><tr><th scope="col">Seed</th><th scope="col">Gate</th><th scope="col">Model</th>'
        '<th scope="col">Oracle</th><th scope="col">Oracle minus model</th>'
        f'<th scope="col">Out-of-sample rows</th></tr></thead><tbody>{body}</tbody></table>'
    )


def seeds_section(summaries: Sequence[SignalSummary]) -> str:
    tables = "".join(_seed_table(s) for s in summaries)
    return (
        '<section id="control-seeds"><h2>Control experiment, version 2, per-seed results</h2>'
        "<p>Every row of the version 2 control data, rounded to three decimals. The version 3 rows are in "
        "<code>docs/data/detection_curve_v3.csv</code>. Model and oracle figures are "
        "out-of-sample balanced accuracy.</p>"
        f'{tables}<p class="source">Source: <code>docs/data/detection_curve.csv</code></p></section>'
    )


RECOMMENDATIONS: tuple[str, ...] = ("BUY", "SELL", "AVOID")
DESIGN_EFFECTS: tuple[int, ...] = (1, 2, 3, 5, 10)
BASELINE_LABELS: tuple[tuple[str, str], ...] = (
    ("always_bearish", "Always bearish"),
    ("always_bullish", "Always bullish"),
    ("class_matched_random", "Class-matched random"),
)
EXPLORATORY_FILE = "exploratory_zero_signal_v3.csv"


@dataclass(frozen=True)
class DesignEffectRow:
    effect: float
    effective_n: float
    interval: Interval

    @property
    def contains_chance(self) -> bool:
        return self.interval.contains(CHANCE)


def _range(low: float, high: float) -> str:
    return f"{low:.3f} to {high:.3f}"


def _cell_v3(row: dict[str, str]) -> CellV3:
    return CellV3(
        signal=float(row["signal"]),
        seed=int(row["seed"]),
        gate_passed=GATE_VALUES[row["gate_passed"]],
        model_used=row["model_used"].strip() or None,
        cv_f1_mean=_optional_float(row["cv_f1_mean"]),
        cv_f1_std=_optional_float(row["cv_f1_std"]),
        model_balanced_accuracy=_optional_float(row["model_balanced_accuracy"]),
        oracle_balanced_accuracy=_optional_float(row["oracle_balanced_accuracy"]),
        holdout_brier_ensemble=_optional_float(row["holdout_brier_ensemble"]),
        majority_accuracy=float(row["majority_accuracy"]),
        label_base_rate=float(row["label_base_rate"]),
        n_oos=int(row["n_oos"]),
    )


def read_cells_v3() -> list[CellV3]:
    return [_cell_v3(r) for r in read_csv("detection_curve_v3.csv")]


def _verdict_label(verdict: bool | None, claim: str) -> str:
    if verdict is None:
        return "not evaluated"
    return f"{claim}: met" if verdict else "not met"


def _primary_row(signal: float, result: dict[str, Any]) -> str:
    if signal == 0.0:
        eq = result["equivalence_zero_signal"]
        pct = f"{eq['interval_confidence'] * 100:.0f}%"
        bound = "n/a" if eq["lower"] is None else f"{pct} interval {_range(eq['lower'], eq['upper'])}"
        extra = f"{eq['margin_low']:.3f} to {eq['margin_high']:.3f}"
        n, mean = eq["n"], eq["mean"]
        verdict = _verdict_label(eq["verdict"], "equivalent to 0.50 within 0.03")
    else:
        row = result["superiority"][signal]
        bound = "n/a" if row["lower_bound"] is None else f"lower bound {row['lower_bound']:.3f}"
        extra = "n/a" if row["p_holm"] is None else f"{row['p_holm']:.2g}"
        n, mean = row["n"], row["mean"]
        verdict = _verdict_label(row["verdict"], "above 0.50")
    return (
        f'<tr><td class="n">{signal:.2f}</td><td class="n">{n}</td><td class="n">{_fmt(mean)}</td>'
        f'<td class="n">{bound}</td><td class="n">{extra}</td><td>{verdict}</td></tr>'
    )


def _secondary_row(signal: float, result: dict[str, Any]) -> str:
    gap = result["paired_gap"][signal]
    gate = result["gate"][signal]
    if gap["mean"] is None:
        gap_text = "n/a"
    else:
        gap_text = f"{gap['mean']:.3f} ({_range(gap['lower'], gap['upper'])})"
    return (
        f'<tr><td class="n">{signal:.2f}</td><td class="n">{gap_text}</td>'
        f'<td class="n">{gap["analytic_ceiling"]:.3f}</td>'
        f'<td class="n">{gate["gate_passed"]} of {gate["n_cells"]}</td></tr>'
    )


def _seed_count_text(result: dict[str, Any]) -> str:
    counts = {signal: gate["n_cells"] for signal, gate in result["gate"].items()}
    if len(set(counts.values())) == 1:
        return f"{next(iter(counts.values()))} seeds per signal"
    return "seeds per signal " + ", ".join(f"{s:.2f}: {n}" for s, n in sorted(counts.items()))


def control_v3_section() -> str:
    cells = read_cells_v3()
    # the analysis takes its signal family from the rows it is given, so a signal with no rows would vanish
    absent = sorted(set(PREREG_SIGNALS) - {c.signal for c in cells})
    if absent:
        raise ValueError(f"pre-registered signal strengths with no rows in the data: {absent}")
    result = analyse(cells)
    signals = sorted(result["paired_gap"])
    primary = "".join(_primary_row(s, result) for s in signals)
    secondary = "".join(_secondary_row(s, result) for s in signals)
    delta, alpha = result["delta"], result["alpha"]
    pct = f"{result['equivalence_zero_signal']['interval_confidence'] * 100:.0f}%"
    gap_confidence = next((g["interval_confidence"] for g in result["paired_gap"].values()), 1 - alpha)
    at_chance = sum(1 for c in cells if c.model_balanced_accuracy == CHANCE)
    return (
        '<section id="control-v3"><h2>Control experiment, version 3 (pre-registered)</h2>'
        "<p>This control scores the two-model ensemble, whose LightGBM member is Platt-calibrated, on seeds "
        "disjoint from version 2, on every seed whether or not the gate passed, so the endpoints describe the "
        "estimator and not the system with abstention. "
        "The specification "
        "(<code>docs/prereg/control-v3.md</code>) was released in a public commit holding the whole code tree, "
        "and the sha256 of the specification and its two analysis files was posted in public issue 1, before "
        "the first public commit of the results. The hashes cover those three files only; the generator, "
        "features, model code and split they import are fixed by that release commit, not by the hashes, "
        "except that the cross-validation scoring was corrected after release in package version 0.1.5 "
        "(issue 33), as recorded in <code>docs/prereg/control-v3-deviations.md</code>. "
        f"The run used {_seed_count_text(result)}. At zero signal "
        f"the endpoint is equivalence within a margin: the {pct} interval must lie inside 0.50 plus or minus "
        f"{delta:g} (alpha {alpha:g} for each one-sided test). At each non-zero signal the endpoint is a mean "
        f"above 0.50, tested one-sided at alpha {alpha:g} after a Holm adjustment across the non-zero "
        "signals. Every verdict below is returned by the pre-registered analysis code and none is "
        "recomputed on this page. A verdict of within the margin is not evidence that the mean equals 0.50, "
        "and it is not a leakage test.</p>"
        "<p>Correction of 22 November 2026 (package version 0.1.5): one fold of one cell (signal 0.6, seed 1008) was "
        "single-class in truth and predictions and had scored 1.0 in the cross-validation F1, which lowers that "
        "cell's mean from 0.665139 to 0.565139. No line of the pre-registered analysis output changes; see "
        "<code>docs/prereg/control-v3-deviations.md</code>.</p>"
        f"<p>{at_chance} of {len(cells)} cells score exactly 0.500, which is what a prediction of a single "
        "class scores. Such cells can pull a mean towards 0.50, so the "
        "results describe the thresholded ensemble estimator with the gate not applied; the pipeline's brief "
        "carries the gated single-model prediction. The oracle thresholds the true conditional "
        "mean at the cost; it is not the rule that maximises balanced accuracy, and the analytic ceiling is "
        "for the zero-cost sign problem, so the gaps are descriptive.</p>"
        '<table id="control-v3-primary"><caption>Pre-registered primary endpoints</caption>'
        '<thead><tr><th scope="col">Signal</th><th scope="col">Seeds</th><th scope="col">Mean</th>'
        f'<th scope="col">Interval, or individual one-sided {(1 - alpha) * 100:.0f}% lower bound (not adjusted)</th>'
        '<th scope="col">Margin or Holm-adjusted p</th><th scope="col">Verdict</th></tr></thead>'
        f"<tbody>{primary}</tbody></table>"
        '<table id="control-v3-secondary"><caption>Secondary and descriptive</caption>'
        '<thead><tr><th scope="col">Signal</th>'
        f'<th scope="col">Oracle minus model, paired mean ({gap_confidence * 100:.0f}% interval)</th>'
        '<th scope="col">Analytic ceiling</th><th scope="col">Gate passed</th></tr></thead>'
        f"<tbody>{secondary}</tbody></table>"
        '<p class="source">Source: <code>docs/data/detection_curve_v3.csv</code> and '
        "<code>src/aer/demo/control_v3.py</code></p></section>"
    )


def read_exploratory() -> dict[str, float] | None:
    if not (DATA_DIR / EXPLORATORY_FILE).is_file():
        return None
    return {r["metric"]: float(r["value"]) for r in read_csv(EXPLORATORY_FILE)}


def exploratory_section() -> str:
    m = read_exploratory()
    if m is None:
        return ""
    seeds = round(m["seeds"])
    own_side = "above" if m["per_seed_own_mean"] > CHANCE else "not above"
    if m["pooled_own_low"] <= CHANCE <= m["pooled_own_high"]:
        own_verdict = (
            "The pooled interval contains 0.50. That is consistent with the per-seed excess being a "
            "finite-window property of the per-seed statistic, and it does not identify the cause. The upper "
            f"end of the pooled interval is {m['pooled_own_high'] - CHANCE:.3f} above 0.50; this interval concerns "
            "the pooled balanced accuracy only and does not rule out other forms of dependence."
        )
    else:
        own_verdict = (
            "The pooled interval excludes 0.50, which indicates a dependence between predictions and their own "
            f"labels of about {m['pooled_own'] - CHANCE:.3f}; its cause is not identified."
        )
    unrelated_verdict = ""
    if not (m["pooled_unrelated_low"] <= CHANCE <= m["pooled_unrelated_high"]):
        unrelated_verdict = (
            " The unrelated-series pooled value has an expected value of 0.50 and its interval excludes 0.50 "
            "in this sample; no cause is identified and it is reported as found."
        )
    return (
        '<section id="exploratory-zero-signal"><h2>Exploratory: the zero-signal excess</h2>'
        "<p>This analysis was not pre-registered and changes no verdict.</p>"
        f"<p>At zero signal the per-seed mean balanced accuracy on {seeds} fresh seeds (first seed "
        f"{round(m['first_seed'])}) was {m['per_seed_own_mean']:.4f} (standard error {m['per_seed_own_se']:.4f}), "
        f"{own_side} 0.50. Pooling the confusion counts over seeds gives {m['pooled_own']:.4f} (95% interval "
        f"from resampling whole seeds {m['pooled_own_low']:.4f} to {m['pooled_own_high']:.4f}). The same "
        "predictions scored against the labels of unrelated series give a per-seed mean of "
        f"{m['per_seed_unrelated_mean']:.4f} (standard error {m['per_seed_unrelated_se']:.4f}) and a pooled "
        f"value of {m['pooled_unrelated']:.4f} ({m['pooled_unrelated_low']:.4f} to "
        f"{m['pooled_unrelated_high']:.4f}).</p>"
        f"<p>{own_verdict}{unrelated_verdict}</p>"
        f"<p>The share of up predictions after an up label minus the share after a down label was "
        f"{m['follow_index']:+.4f}; a negative value means up predictions were less frequent after an up label "
        "than after a down label. "
        f"{round(m['single_class_cells'])} of {seeds} cells predicted a single class.</p>"
        f'<p class="source">Source: <code>docs/data/{EXPLORATORY_FILE}</code> and '
        "<code>docs/report/explore_zero_signal_v3.py</code></p></section>"
    )


def wilson_interval(successes: float, n: float, confidence: float = CONFIDENCE) -> Interval:
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half), round(n))


def _tally(rows: Sequence[dict[str, str]]) -> tuple[int, int, int, int]:
    bullish = bullish_correct = bearish = bearish_correct = 0
    for r in rows:
        resolved, correct = int(r["n_resolved"]), int(r["n_synth_correct"])
        if r["recommendation"] == "BUY":
            bullish, bullish_correct = bullish + resolved, bullish_correct + correct
        elif r["recommendation"] in {"SELL", "AVOID"}:
            bearish, bearish_correct = bearish + resolved, bearish_correct + correct
    return bullish, bullish_correct, bearish, bearish_correct


def outcome_counts(rows: Sequence[dict[str, str]]) -> tuple[int, int]:
    """Resolved actionable calls followed by an up move and by a down move.

    Recorded calls are scored by direction at zero: a BUY is correct on an up move and a SELL or AVOID on a down
    move, with no cost band. A wrong BUY therefore fixes the outcome as down and a wrong SELL or AVOID as up.
    """
    b, bc, s, sc = _tally(rows)
    if b + s == 0:
        raise ValueError("no resolved actionable calls")
    return bc + (s - sc), sc + (b - bc)


def baselines(rows: Sequence[dict[str, str]]) -> dict[str, float]:
    """Accuracy of each simple rule on the resolved actionable calls, exact under the recorded scoring rule."""
    b, _, s, _ = _tally(rows)
    up, down = outcome_counts(rows)
    n = b + s
    return {
        "always_bearish": down / n,
        "always_bullish": up / n,
        "class_matched_random": (b / n) * (up / n) + (s / n) * (down / n),
    }


def design_effect_rows(correct: float, n: float, effects: Sequence[float] = DESIGN_EFFECTS) -> list[DesignEffectRow]:
    p = correct / n
    rows: list[DesignEffectRow] = []
    for effect in effects:
        effective = n / effect
        rows.append(DesignEffectRow(effect, effective, wilson_interval(p * effective, effective)))
    return rows


def _recommendation_row(rec: str, row: dict[str, str]) -> str:
    resolved, correct = int(row["n_resolved"]), int(row["n_synth_correct"])
    if resolved == 0:
        return (
            f'<tr><td>{rec}</td><td class="n">0</td><td class="n">0</td>'
            '<td class="n">n/a</td><td class="n">n/a</td></tr>'
        )
    interval = wilson_interval(correct, resolved)
    return (
        f'<tr><td>{rec}</td><td class="n">{resolved:,}</td><td class="n">{correct:,}</td>'
        f'<td class="n">{correct / resolved:.3f}</td>'
        f'<td class="n">{_range(interval.low, interval.high)}</td></tr>'
    )


def baselines_section() -> str:
    rows = read_csv("eval_recommendation.csv")
    b, bc, s, sc = _tally(rows)
    n, correct = b + s, bc + sc
    accuracy = correct / n
    interval = wilson_interval(correct, n)
    rates = baselines(rows)
    up, down = outcome_counts(rows)
    confidence = f"{CONFIDENCE:.0%}"
    counts = {"always_bearish": f"{down:,}", "always_bullish": f"{up:,}", "class_matched_random": "n/a"}
    baseline_rows = f'<tr><td>Recorded calls</td><td class="n">{correct:,}</td><td class="n">{accuracy:.3f}</td></tr>'
    baseline_rows += "".join(
        f'<tr><td>{label}</td><td class="n">{counts[key]}</td><td class="n">{rates[key]:.3f}</td></tr>'
        for key, label in BASELINE_LABELS
    )
    bearish = rates["always_bearish"]
    if accuracy < bearish:
        verdict = (
            f"The recorded accuracy of {accuracy:.3f} is below the always-bearish rule ({bearish:.3f}): on these "
            "calls, always saying down scores higher. The two differ only on the "
            f"{b:,} BUY calls, of which {b - bc:,} were followed by a down move."
        )
    else:
        verdict = (
            f"The recorded accuracy of {accuracy:.3f} is not below the always-bearish rule ({bearish:.3f}) on "
            "these calls."
        )
    effect_rows = "".join(
        f'<tr><td class="n">{r.effect:g}</td><td class="n">{r.effective_n:.1f}</td>'
        f'<td class="n">{_range(r.interval.low, r.interval.high)}</td>'
        f"<td>{'yes' if r.contains_chance else 'no'}</td></tr>"
        for r in design_effect_rows(correct, n)
    )
    by_row = {r["recommendation"]: r for r in rows}
    rec_rows = "".join(_recommendation_row(rec, by_row[rec]) for rec in RECOMMENDATIONS if rec in by_row)
    return (
        '<section id="evaluation-baselines"><h2>Recorded calls against simple baselines</h2>'
        f"<p>Recorded accuracy is {correct:,} of {n:,} ({accuracy:.3f}), with a {confidence} Wilson interval of "
        f"{_range(interval.low, interval.high)}. The interval assumes independent calls.</p>"
        '<table id="by-recommendation"><caption>Accuracy by recommendation</caption>'
        '<thead><tr><th scope="col">Recommendation</th><th scope="col">Resolved</th><th scope="col">Correct</th>'
        f'<th scope="col">Accuracy</th><th scope="col">{confidence} Wilson interval</th></tr></thead>'
        f"<tbody>{rec_rows}</tbody></table>"
        '<table id="baselines"><caption>Simple rules on the same resolved calls</caption>'
        '<thead><tr><th scope="col">Rule</th><th scope="col">Correct</th><th scope="col">Accuracy</th>'
        f"</tr></thead><tbody>{baseline_rows}</tbody></table>"
        "<p>Recorded calls are scored by direction: the outcome is up when the five-day return is above zero "
        "and down otherwise, a BUY is correct on an up move and a SELL or AVOID on a down move "
        "(<code>src/aer/evaluation/weekly_scorer.py</code>). There is no cost band, so a wrong call fixes the "
        f"direction of the move and the published counts give each rule exactly: {down:,} of the {n:,} resolved "
        f"calls were followed by a down move and {up:,} by an up move. The class-matched random rule says "
        "bullish with the recorded share of bullish calls, independently of the outcome, and its figure is an "
        f"expected accuracy. {verdict} The comparison is descriptive: the calls are dependent, as the next "
        "table explains, and the direction split is that of one period.</p>"
        '<table id="design-effect"><caption>Sensitivity of the accuracy interval to dependence between '
        "calls</caption>"
        '<thead><tr><th scope="col">Design effect</th><th scope="col">Effective calls</th>'
        f'<th scope="col">{confidence} Wilson interval at the assumed design effect</th>'
        '<th scope="col">Contains 0.50</th></tr></thead>'
        f"<tbody>{effect_rows}</tbody></table>"
        "<p>Calls on the same ticker overlap in their five-day windows and calls on the same day share the "
        "market move, so the independent count is lower than the nominal count. Dividing the nominal count by "
        "a design effect d gives the effective count. The published data carry no ticker or date, so the "
        "effect cannot be "
        "estimated here; the rows are assumed scenarios, not a measured range. The 0.50 line is also not "
        "the no-skill rate for these calls, which the baselines above describe.</p>"
        '<p class="source">Source: <code>docs/data/eval_recommendation.csv</code></p></section>'
    )


LIMITS: tuple[str, ...] = (
    (
        "<li>The control experiments use one synthetic signal family. They show the harness can find that "
        "signal and say nothing about whether real prices carry one.</li>"
    ),
    (
        "<li>Version 2 of the control measures one uncalibrated classifier per seed. Version 3 measures the "
        "two-model ensemble with the gate's abstention not applied, and only its LightGBM member is "
        "Platt-calibrated.</li>"
    ),
    (
        "<li>LightGBM row subsampling was never active. Earlier releases set <code>subsample</code> to 0.8 "
        "with no <code>subsample_freq</code>, which LightGBM ignores, so no fit made by this code, the "
        "pre-registered control included, used row subsampling.</li>"
    ),
    (
        "<li>Recorded calls are scored by direction at zero on adjusted closes, while the ML label is a "
        "five-day return above the round-trip cost on unadjusted closes, so the recorded ML accuracy and "
        "the training target measure different quantities.</li>"
    ),
    ("<li>Briefs are scored against a five-day move, while the underlying thesis spans weeks to " "months.</li>"),
    (
        "<li>Calls on the same ticker or day are dependent, and the published data carry no ticker or date, "
        "so the dependence cannot be measured here; the design-effect table shows the interval under assumed "
        "levels of it.</li>"
    ),
    (
        "<li>The repository README lists further limits, and open defects are tracked as public GitHub issues "
        "labelled known-limitation.</li>"
    ),
)


def limits_section() -> str:
    return f'<section id="limits"><h2>Limits</h2><ul>{"".join(LIMITS)}</ul></section>'


def figures_section() -> str:
    items = "".join(
        f'<figure><img src="img/{name}" alt="{html.escape(alt)}" width="680">'
        f"<figcaption>{html.escape(alt)}.</figcaption></figure>"
        for name, alt in FIGURES
    )
    return f'<section id="figures"><h2>Figures</h2>{items}</section>'


def footer() -> str:
    commit = os.environ.get("GITHUB_SHA", "")[:12] or "local build"
    data = "".join(
        f"<tr><td><code>docs/data/{p.name}</code></td><td><code>{sha256(p)[:16]}</code></td></tr>"
        for p in sorted(DATA_DIR.iterdir())
        if p.is_file()
    )
    return (
        "<footer><h2>Reproducibility</h2>"
        f"<p>Commit: <code>{html.escape(commit)}</code>. Evaluation spec <code>src/aer/evaluation/spec.py</code> "
        f"sha256 <code>{sha256(SPEC_FILE)[:16]}</code>. Built with Python {platform.python_version()}.</p>"
        '<table><caption>Input data (sha256, first 16 characters)</caption><thead><tr><th scope="col">File</th>'
        f'<th scope="col">sha256</th></tr></thead><tbody>{data}</tbody></table></footer>'
    )


def render() -> str:
    summaries = control()
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Agentic equity research: evaluation report</title>"
        f"<style>{CSS}</style></head><body><main>"
        "<h1>Agentic equity research: evaluation report</h1>"
        "<p>Recorded calls are published as whole-sample aggregates only, with no ticker, week or date "
        "dimension; the synthetic control is also shown per seed. "
        "No figure here is a trading result.</p>"
        f"{cohort_section()}{baselines_section()}{detection_section(summaries)}{control_v3_section()}"
        f"{exploratory_section()}{seeds_section(summaries)}{figures_section()}{limits_section()}</main>"
        f"{footer()}"
        "</body></html>\n"
    )


def build(out: Path = DEFAULT_OUT) -> Path:
    (out / "img").mkdir(parents=True, exist_ok=True)
    for name, _ in FIGURES:
        shutil.copyfile(IMG_DIR / name, out / "img" / name)
    index = out / "index.html"
    index.write_text(render(), encoding="utf-8", newline="\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    print(build(parser.parse_args().out))


if __name__ == "__main__":
    main()
