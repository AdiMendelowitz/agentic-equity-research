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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
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
        f"{cohort_section()}{detection_section(summaries)}{seeds_section(summaries)}{figures_section()}</main>"
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
