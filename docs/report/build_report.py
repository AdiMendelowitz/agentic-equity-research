"""Build the static evaluation report (one HTML page plus figures) from the committed data.

Every number on the page is computed here from a file under docs/data, and the footer records the
hash of each input so a reader can check which data produced the page.
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
        "Model and oracle balanced accuracy against injected signal strength, with the analytic ceiling "
        "and the 0.50 chance line",
    ),
    ("calibration.svg", "Hit rate by conviction bin with 95% Wilson intervals, against the 0.50 line"),
    ("recommendation_hit_rate.svg", "Hit rate by recommendation with 95% Wilson intervals"),
    ("ml_gate.svg", "Share of rows on which the ML gate abstained, and its accuracy where it predicted"),
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
        CohortStep("Briefs in the evaluation store", briefs),
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
    return GateComparison(first, last, float(p_value))


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


def cohort_section() -> str:
    steps = cohort()
    resolved, correct = steps[4].count, steps[5].count
    body = "".join(f'<tr><td>{html.escape(s.label)}</td><td class="n">{s.count:,}</td></tr>' for s in steps)
    return (
        '<section id="cohort"><h2>Evaluation cohort</h2>'
        "<p>Accuracy is measured on resolved actionable calls only. Each step below is computed from the data "
        "file named under the table.</p>"
        f'<table><caption>From all briefs to the scored cohort</caption><thead><tr><th scope="col">Step</th>'
        f'<th scope="col">Rows</th></tr></thead><tbody>{body}</tbody></table>'
        f"<p>Accuracy on the scored cohort: {correct:,} of {resolved:,} ({correct / resolved:.3f}).</p>"
        '<p class="source">Source: <code>docs/data/eval_recommendation.csv</code></p></section>'
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
        f"and on {last.gate_predicted} of {len(last.seeds)} at signal {last.signal} (Fisher exact test, "
        f"two-sided p = {comparison.p_value:.3f}). The gate is reported as a failed component.</p>"
    )


def detection_section(summaries: Sequence[SignalSummary]) -> str:
    rows = "".join(_summary_row(s) for s in summaries)
    return (
        '<section id="control"><h2>Control experiment</h2>'
        "<p>Synthetic prices with a known injected signal; no market data. Seeds are treated as independent "
        "replications. Every interval is a 95% Student's t interval over seeds "
        f"({_t_note(summaries)}). The oracle-minus-model gap is computed within each seed, so it is paired.</p>"
        "<p>The expectations fixed before the run, in <code>src/aer/demo/control.py</code>, use a normal "
        "multiplier of 1.96, and the README table reports those narrower intervals. This page reports t "
        "intervals; the column on chance is computed from them.</p>"
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
        f'<table id="seeds-{summary.signal}"><caption>Per-seed results at signal {summary.signal}</caption>'
        '<thead><tr><th scope="col">Seed</th><th scope="col">Gate</th><th scope="col">Model</th>'
        '<th scope="col">Oracle</th><th scope="col">Oracle minus model</th>'
        f'<th scope="col">Out-of-sample rows</th></tr></thead><tbody>{body}</tbody></table>'
    )


def seeds_section(summaries: Sequence[SignalSummary]) -> str:
    tables = "".join(_seed_table(s) for s in summaries)
    return (
        '<section id="control-seeds"><h2>Control experiment: per-seed results</h2>'
        "<p>Every row of the control data, rounded to three decimals. Model and oracle figures are "
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
        verdict = _verdict_label(eq["verdict"], "equivalent to chance")
    else:
        row = result["superiority"][signal]
        bound = "n/a" if row["lower_bound"] is None else f"lower bound {row['lower_bound']:.3f}"
        extra = "n/a" if row["p_holm"] is None else f"{row['p_holm']:.2g}"
        n, mean = row["n"], row["mean"]
        verdict = _verdict_label(row["verdict"], "above chance")
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
    result = analyse(read_cells_v3())
    signals = sorted(result["paired_gap"])
    primary = "".join(_primary_row(s, result) for s in signals)
    secondary = "".join(_secondary_row(s, result) for s in signals)
    delta, alpha = result["delta"], result["alpha"]
    pct = f"{result['equivalence_zero_signal']['interval_confidence'] * 100:.0f}%"
    gap_confidence = next((g["interval_confidence"] for g in result["paired_gap"].values()), 1 - alpha)
    return (
        '<section id="control-v3"><h2>Control experiment, version 3 (pre-registered)</h2>'
        "<p>This control scores the deployed calibrated ensemble on seeds disjoint from version 2. The "
        "specification (<code>docs/prereg/control-v3.md</code>) was published, and its sha256 posted in public "
        f"issue 1 of the repository, before the run. The run used {_seed_count_text(result)}. At zero signal "
        f"the endpoint is equivalence to chance: the {pct} interval must lie inside 0.50 plus or minus "
        f"{delta:g} (alpha {alpha:g} for each one-sided test). At each non-zero signal the endpoint is a mean "
        f"above 0.50, tested one-sided at alpha {alpha:g} after a Holm adjustment across the non-zero "
        "signals. Every verdict below is returned by the pre-registered analysis code and none is "
        "recomputed on this page.</p>"
        '<table id="control-v3-primary"><caption>Pre-registered primary endpoints</caption>'
        '<thead><tr><th scope="col">Signal</th><th scope="col">Seeds</th><th scope="col">Mean</th>'
        '<th scope="col">Interval or one-sided lower bound</th>'
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
            "The pooled interval contains 0.50, so most of the per-seed excess is a property of the per-seed "
            "statistic in a finite window; any dependence between predictions and their own labels is below "
            f"{m['pooled_own_high'] - CHANCE:.3f}."
        )
    else:
        own_verdict = (
            "The pooled interval excludes 0.50, which indicates a dependence between predictions and their own "
            f"labels of about {m['pooled_own'] - CHANCE:.3f}; its cause is not identified."
        )
    unrelated_verdict = ""
    if not (m["pooled_unrelated_low"] <= CHANCE <= m["pooled_unrelated_high"]):
        unrelated_verdict = (
            " The unrelated-series pooled value should be 0.50 by construction and its interval excludes it; "
            "this is unexplained and is reported as found."
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
        f"{m['follow_index']:+.4f}; a negative value means predictions ran against the last completed label. "
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


def baseline_bounds(rows: Sequence[dict[str, str]]) -> dict[str, tuple[float, float]]:
    b, bc, s, sc = _tally(rows)
    n = b + s
    if n == 0:
        raise ValueError("no resolved actionable calls")
    wrong_buys, wrong_sells = b - bc, s - sc
    return {
        "always_bearish": (sc / n, (sc + wrong_buys) / n),
        "always_bullish": (bc / n, (bc + wrong_sells) / n),
        "class_matched_random": (
            (b / n) * (bc / n) + (s / n) * (sc / n),
            (b / n) * ((bc + wrong_sells) / n) + (s / n) * ((sc + wrong_buys) / n),
        ),
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
    bounds = baseline_bounds(rows)
    confidence = f"{CONFIDENCE:.0%}"
    bound_rows = (
        f'<tr><td>Recorded calls</td><td class="n">{accuracy:.3f}</td>' f'<td class="n">{accuracy:.3f}</td></tr>'
    )
    bound_rows += "".join(
        f'<tr><td>{label}</td><td class="n">{bounds[key][0]:.3f}</td>' f'<td class="n">{bounds[key][1]:.3f}</td></tr>'
        for key, label in BASELINE_LABELS
    )
    bearish_upper = bounds["always_bearish"][1]
    if accuracy < bearish_upper:
        verdict = (
            f"The recorded accuracy of {accuracy:.3f} is below the upper bound of the always-bearish baseline "
            f"({bearish_upper:.3f}), so the published counts cannot rule out that an always-bearish rule "
            "scores at least as well."
        )
    else:
        verdict = (
            f"The recorded accuracy of {accuracy:.3f} is not below the upper bound of the always-bearish "
            f"baseline ({bearish_upper:.3f}), so an always-bearish rule cannot score higher."
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
        '<table id="baselines"><caption>Baselines, as bounds computable from the published counts</caption>'
        '<thead><tr><th scope="col">Rule</th><th scope="col">Lower bound</th><th scope="col">Upper bound</th>'
        f"</tr></thead><tbody>{bound_rows}</tbody></table>"
        "<p>The published counts give bounds, not point values, because an incorrect call does not reveal "
        "whether the move was in the opposite direction or inside the cost band. Exact baselines need one "
        "further whole-sample aggregate: the counts of forward returns above the cost, below minus the cost "
        "and inside the band. That aggregate carries no ticker, week or date dimension. "
        f"{verdict}</p>"
        '<table id="design-effect"><caption>Sensitivity of the accuracy interval to dependence between '
        "calls</caption>"
        '<thead><tr><th scope="col">Design effect</th><th scope="col">Effective calls</th>'
        f'<th scope="col">{confidence} interval</th><th scope="col">Contains 0.50</th></tr></thead>'
        f"<tbody>{effect_rows}</tbody></table>"
        "<p>Calls on the same ticker overlap in their five-day windows and calls on the same day share the "
        "market move, so the independent count is lower than the nominal count. A design effect of d divides "
        "the effective count by d. The published data carry no ticker or date, so the effect cannot be "
        "estimated here and the table shows the range.</p>"
        '<p class="source">Source: <code>docs/data/eval_recommendation.csv</code></p></section>'
    )


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
        "<p>Whole-sample aggregates only; no ticker, week or date dimension is published. "
        "No figure here is a trading result.</p>"
        f"{cohort_section()}{baselines_section()}{detection_section(summaries)}{control_v3_section()}"
        f"{exploratory_section()}{seeds_section(summaries)}{figures_section()}</main>"
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
