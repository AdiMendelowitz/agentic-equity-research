from __future__ import annotations

import csv
import importlib.util
import math
import re
import statistics
import sys
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aer.demo.control_v3 import CellV3, analyse  # noqa: E402

DATA_DIR = ROOT / "docs" / "data"
T_975_19 = 2.093024
V3_HEADER = (
    "signal,seed,gate_passed,model_used,cv_f1_mean,cv_f1_std,model_balanced_accuracy,"
    "oracle_balanced_accuracy,holdout_brier_ensemble,majority_accuracy,label_base_rate,n_oos"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_report", ROOT / "docs" / "report" / "build_report.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_report = _load()


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lang: str | None = None
        self.headings: list[int] = []
        self.images: list[dict[str, str | None]] = []
        self.external: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "html":
            self.lang = a.get("lang")
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append(int(tag[1]))
        if tag == "img":
            self.images.append(a)
        for key in ("src", "href"):
            value = a.get(key) or ""
            if tag in {"img", "script", "link"} and re.match(r"(https?:)?//", value):
                self.external.append(value)


class _Tables(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: dict[str, list[list[str]]] = {}
        self._table: str | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table = dict(attrs).get("id") or None
            if self._table is not None:
                self.tables[self._table] = []
        elif self._table is not None and tag == "tr":
            self.tables[self._table].append([])
        elif self._table is not None and tag == "td":
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._table is not None and self._cell is not None:
            self.tables[self._table][-1].append("".join(self._cell))
            self._cell = None
        elif tag == "table":
            self._table = None


def _control_rows() -> list[dict[str, str]]:
    with (DATA_DIR / "detection_curve.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _signal(row: dict[str, str]) -> str:
    return f"{float(row['signal']):.2f}"


def _tables(site: Path) -> dict[str, list[list[str]]]:
    parser = _Tables()
    parser.feed((site / "index.html").read_text(encoding="utf-8"))
    return {name: [row for row in rows if row] for name, rows in parser.tables.items()}


def _fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    n1, n2, k = a + b, c + d, a + c

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)

    observed = prob(a)
    support = range(max(0, k - n2), min(n1, k) + 1)
    return sum(prob(x) for x in support if prob(x) <= observed * (1 + 1e-9))


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site")
    build_report.build(out)
    return out


@pytest.fixture(scope="module")
def page(site: Path) -> _Page:
    parsed = _Page()
    parsed.feed((site / "index.html").read_text(encoding="utf-8"))
    return parsed


def test_build_is_byte_identical_across_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    first = build_report.build(tmp_path / "a").read_bytes()
    second = build_report.build(tmp_path / "b").read_bytes()
    assert first == second


def test_cohort_counts_trace_to_the_data_file() -> None:
    with (DATA_DIR / "eval_recommendation.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actionable = [r for r in rows if r["recommendation"] != "HOLD"]
    expected = [
        sum(int(r["n_rows"]) for r in rows),
        sum(int(r["n_rows"]) for r in rows if r["recommendation"] == "HOLD"),
        sum(int(r["n_rows"]) for r in actionable),
        sum(int(r["n_rows"]) - int(r["n_resolved"]) for r in actionable),
        sum(int(r["n_resolved"]) for r in actionable),
        sum(int(r["n_synth_correct"]) for r in actionable),
    ]
    steps = build_report.cohort()
    assert [s.count for s in steps] == expected
    assert steps[0].count == steps[1].count + steps[2].count
    assert steps[2].count == steps[3].count + steps[4].count


def test_every_cohort_count_is_on_the_page(site: Path) -> None:
    text = (site / "index.html").read_text(encoding="utf-8")
    for step in build_report.cohort():
        assert f'<td class="n">{step.count:,}</td>' in text


def test_page_structure_is_accessible(page: _Page) -> None:
    assert page.lang == "en"
    assert page.headings.count(1) == 1
    assert page.headings[0] == 1
    assert all(b - a <= 1 for a, b in zip(page.headings, page.headings[1:]))


def test_every_image_has_alt_text_and_ships_with_the_page(site: Path, page: _Page) -> None:
    assert len(page.images) == len(build_report.FIGURES)
    for img in page.images:
        assert (img.get("alt") or "").strip()
        assert (site / str(img["src"])).is_file()


def test_page_loads_nothing_from_another_origin(page: _Page) -> None:
    assert page.external == []


def test_footer_hashes_every_data_file(site: Path) -> None:
    text = (site / "index.html").read_text(encoding="utf-8")
    for path in sorted(p for p in DATA_DIR.iterdir() if p.is_file()):
        assert f"<code>docs/data/{path.name}</code>" in text
        assert build_report.sha256(path)[:16] in text


def test_commit_comes_from_the_ci_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef0123")
    text = build_report.build(tmp_path).read_text(encoding="utf-8")
    assert "<code>0123456789ab</code>" in text


def test_t_critical_value_for_twenty_seeds_is_pinned() -> None:
    assert build_report.t_critical(20) == pytest.approx(T_975_19, abs=5e-7)


def test_t_interval_needs_at_least_two_values() -> None:
    assert build_report.t_interval([0.5]) is None


def test_interval_half_width_is_t_times_sd_over_root_n() -> None:
    rows = _control_rows()
    summaries = build_report.control()
    assert [s.signal for s in summaries] == sorted({_signal(r) for r in rows}, key=float)
    for summary in summaries:
        group = [r for r in rows if _signal(r) == summary.signal]
        model = [float(r["model_balanced_accuracy"]) for r in group]
        oracle = [float(r["oracle_balanced_accuracy"]) for r in group]
        gap = [o - m for o, m in zip(oracle, model, strict=True)]
        for interval, values in ((summary.model, model), (summary.oracle, oracle), (summary.gap, gap)):
            assert interval is not None
            assert interval.n == len(values) == 20
            half = T_975_19 * statistics.stdev(values) / math.sqrt(len(values))
            assert interval.mean == pytest.approx(statistics.fmean(values), rel=1e-12, abs=1e-12)
            assert interval.high - interval.mean == pytest.approx(half, rel=1e-6, abs=1e-12)
            assert interval.mean - interval.low == pytest.approx(half, rel=1e-6, abs=1e-12)


def test_every_per_seed_value_on_the_page_equals_the_csv(site: Path) -> None:
    tables = _tables(site)
    rows = _control_rows()
    signals = sorted({_signal(r) for r in rows}, key=float)
    assert sorted(name for name in tables if name.startswith("seeds-")) == sorted(f"seeds-{s}" for s in signals)
    for signal in signals:
        group = sorted((r for r in rows if _signal(r) == signal), key=lambda r: int(r["seed"]))
        expected = [
            [
                r["seed"],
                "predicted" if r["gate_predicted"] == "True" else "abstained",
                f"{float(r['model_balanced_accuracy']):.3f}",
                f"{float(r['oracle_balanced_accuracy']):.3f}",
                f"{float(r['oracle_balanced_accuracy']) - float(r['model_balanced_accuracy']):.3f}",
                r["n_oos"],
            ]
            for r in group
        ]
        assert tables[f"seeds-{signal}"] == expected


def test_summary_rows_follow_the_csv(site: Path) -> None:
    rows = _control_rows()
    signals = sorted({_signal(r) for r in rows}, key=float)
    summary_rows = _tables(site)["control-summary"]
    assert [row[0] for row in summary_rows] == signals
    for row, signal in zip(summary_rows, signals, strict=True):
        group = [r for r in rows if _signal(r) == signal]
        model = [float(r["model_balanced_accuracy"]) for r in group]
        centre = statistics.fmean(model)
        half = T_975_19 * statistics.stdev(model) / math.sqrt(len(model))
        predicted = sum(1 for r in group if r["gate_predicted"] == "True")
        assert row[1] == str(len(model))
        assert row[2] == f"{centre:.3f} ({centre - half:.3f} to {centre + half:.3f})"
        assert row[3] == ("yes" if centre - half <= 0.5 <= centre + half else "no")
        assert row[6] == f"{predicted} of {len(group)}"


def test_gate_comparison_matches_an_exact_hypergeometric_sum(site: Path) -> None:
    rows = _control_rows()
    signals = sorted({_signal(r) for r in rows}, key=float)
    counts: list[tuple[int, int]] = []
    for signal in (signals[0], signals[-1]):
        group = [r for r in rows if _signal(r) == signal]
        predicted = sum(1 for r in group if r["gate_predicted"] == "True")
        counts.append((predicted, len(group) - predicted))
    expected = _fisher_two_sided(*counts[0], *counts[1])
    comparison = build_report.gate_comparison(build_report.control())
    assert comparison is not None
    assert comparison.p_value == pytest.approx(expected, rel=1e-9)
    assert f"gives p = {expected:.3f}" in (site / "index.html").read_text(encoding="utf-8")


def test_gate_comparison_pairs_seeds_with_an_exact_mcnemar_test(site: Path) -> None:
    rows = _control_rows()
    signals = sorted({_signal(r) for r in rows}, key=float)
    by_seed = [
        {r["seed"]: r["gate_predicted"] == "True" for r in rows if _signal(r) == s} for s in (signals[0], signals[-1])
    ]
    only_first = sum(1 for seed, a in by_seed[0].items() if a and not by_seed[1][seed])
    only_last = sum(1 for seed, a in by_seed[0].items() if by_seed[1][seed] and not a)
    n = only_first + only_last
    tail = sum(math.comb(n, k) for k in range(min(only_first, only_last) + 1)) / 2**n
    expected = min(1.0, 2 * tail)
    comparison = build_report.gate_comparison(build_report.control())
    assert comparison is not None
    assert comparison.discordant == (only_first, only_last)
    assert comparison.paired_p_value == pytest.approx(expected, rel=1e-9)
    assert f"exact McNemar test, two-sided p = {expected:.3f}" in (site / "index.html").read_text(encoding="utf-8")


def test_unexpected_gate_value_is_rejected() -> None:
    with pytest.raises(KeyError):
        build_report._seed_row(
            {
                "seed": "0",
                "gate_predicted": "true",
                "model_balanced_accuracy": "0.5",
                "oracle_balanced_accuracy": "0.5",
                "n_oos": "1",
            }
        )


def _eval_rows() -> list[dict[str, str]]:
    with (DATA_DIR / "eval_recommendation.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _optional(value: str) -> float | None:
    return float(value) if value.strip() else None


def _cells_v3() -> list[CellV3]:
    with (DATA_DIR / "detection_curve_v3.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        CellV3(
            signal=float(r["signal"]),
            seed=int(r["seed"]),
            gate_passed=r["gate_passed"] == "True",
            model_used=r["model_used"] or None,
            cv_f1_mean=_optional(r["cv_f1_mean"]),
            cv_f1_std=_optional(r["cv_f1_std"]),
            model_balanced_accuracy=_optional(r["model_balanced_accuracy"]),
            oracle_balanced_accuracy=_optional(r["oracle_balanced_accuracy"]),
            holdout_brier_ensemble=_optional(r["holdout_brier_ensemble"]),
            majority_accuracy=float(r["majority_accuracy"]),
            label_base_rate=float(r["label_base_rate"]),
            n_oos=int(r["n_oos"]),
        )
        for r in rows
    ]


def _verdict_text(zero_signal: bool, verdict: bool | None) -> str:
    if verdict is None:
        return "not evaluated"
    claim = "equivalent to 0.50 within 0.03" if zero_signal else "above 0.50"
    return f"{claim}: met" if verdict else "not met"


def _exploratory_csv(directory: Path, low: float, high: float) -> None:
    metrics = {
        "seeds": 40,
        "first_seed": 2000,
        "per_seed_own_mean": 0.5123,
        "per_seed_own_se": 0.0041,
        "per_seed_unrelated_mean": 0.5087,
        "per_seed_unrelated_se": 0.0039,
        "pooled_own": 0.5010,
        "pooled_own_low": low,
        "pooled_own_high": high,
        "pooled_unrelated": 0.5000,
        "pooled_unrelated_low": 0.4950,
        "pooled_unrelated_high": 0.5050,
        "single_class_cells": 3,
        "follow_index": 1,
    }
    lines = ["metric,value"] + [f"{key},{value}" for key, value in metrics.items()]
    (directory / "exploratory_zero_signal_v3.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_wilson_interval_matches_hand_values() -> None:
    mid = build_report.wilson_interval(50, 100)
    assert mid.mean == pytest.approx(0.5)
    assert mid.low == pytest.approx(0.4038, abs=5e-4)
    assert mid.high == pytest.approx(0.5962, abs=5e-4)
    edge = build_report.wilson_interval(0, 10)
    assert edge.low == pytest.approx(0.0, abs=1e-12)
    assert edge.high == pytest.approx(0.2775, abs=5e-4)


def test_outcome_counts_follow_from_the_directional_scoring_rule() -> None:
    by_rec = {r["recommendation"]: (int(r["n_resolved"]), int(r["n_synth_correct"])) for r in _eval_rows()}
    buy_n, buy_ok = by_rec["BUY"]
    sell_n = by_rec["SELL"][0] + by_rec["AVOID"][0]
    sell_ok = by_rec["SELL"][1] + by_rec["AVOID"][1]
    up, down = build_report.outcome_counts(_eval_rows())
    # a correct BUY and a wrong SELL or AVOID are up moves; a correct SELL or AVOID and a wrong BUY are down moves
    assert up == buy_ok + (sell_n - sell_ok)
    assert down == sell_ok + (buy_n - buy_ok)
    assert up + down == buy_n + sell_n


def test_baselines_match_the_committed_counts() -> None:
    assert build_report.outcome_counts(_eval_rows()) == (279, 350)
    rates = build_report.baselines(_eval_rows())
    assert set(rates) == {"always_bearish", "always_bullish", "class_matched_random"}
    assert rates["always_bearish"] == pytest.approx(350 / 629)
    assert rates["always_bullish"] == pytest.approx(279 / 629)
    assert rates["always_bearish"] + rates["always_bullish"] == pytest.approx(1.0)
    assert rates["class_matched_random"] == pytest.approx((244 / 629) * (279 / 629) + (385 / 629) * (350 / 629))


def test_correction_note_matches_the_committed_data(site: Path) -> None:
    steps = build_report.cohort()
    used = build_report.CORRECTION_ROWS[1]
    assert (used[1], used[2]) == (steps[5].count, steps[4].count)
    gate = list(csv.DictReader((DATA_DIR / "eval_ml_gate.csv").read_text(encoding="utf-8").splitlines()))
    assert (used[3], used[4]) == (
        sum(int(r["n_ml_correct"]) for r in gate),
        sum(int(r["n_ml_resolved"]) for r in gate),
    )
    rows = _tables(site)["correction"]
    assert [r[0] for r in rows] == [rule for rule, *_ in build_report.CORRECTION_ROWS]
    assert rows[1][1] == f"{steps[5].count} of {steps[4].count} ({steps[5].count / steps[4].count:.3f})"


def test_the_scoring_rule_the_baselines_assume_is_the_recorded_one() -> None:
    from aer.evaluation.eval_models import compute_synth_correct

    assert compute_synth_correct("BUY", "UP") is True and compute_synth_correct("BUY", "DOWN") is False
    for bearish in ("SELL", "AVOID"):
        assert compute_synth_correct(bearish, "DOWN") is True and compute_synth_correct(bearish, "UP") is False
    assert compute_synth_correct("HOLD", "UP") is None
    scorer = (ROOT / "src" / "aer" / "evaluation" / "weekly_scorer.py").read_text(encoding="utf-8")
    assert 'actual_direction = "UP" if actual_return > 0 else "DOWN"' in scorer


def test_outcome_counts_need_resolved_calls() -> None:
    empty = [{"recommendation": "BUY", "n_rows": "3", "n_resolved": "0", "n_synth_correct": "0"}]
    with pytest.raises(ValueError):
        build_report.outcome_counts(empty)


def test_design_effect_one_reproduces_the_plain_wilson_interval() -> None:
    plain = build_report.wilson_interval(314, 629)
    first = build_report.design_effect_rows(314, 629)[0]
    assert first.effect == 1
    assert first.interval.low == pytest.approx(plain.low, abs=1e-12)
    assert first.interval.high == pytest.approx(plain.high, abs=1e-12)


def test_design_effect_widens_the_interval_and_divides_the_count() -> None:
    rows = build_report.design_effect_rows(314, 629)
    assert [r.effect for r in rows] == [1, 2, 3, 5, 10]
    widths = [r.interval.high - r.interval.low for r in rows]
    assert all(a < b for a, b in zip(widths, widths[1:]))
    for row in rows:
        assert row.effective_n == pytest.approx(629 / row.effect)
        assert row.contains_chance == (row.interval.low <= 0.5 <= row.interval.high)


def test_baselines_table_shows_exact_counts_and_rates(site: Path) -> None:
    rows = _tables(site)["baselines"]
    assert rows[0] == ["Recorded calls", "314", f"{314 / 629:.3f}"]
    assert rows[1] == ["Always bearish", "350", f"{350 / 629:.3f}"]
    assert rows[2] == ["Always bullish", "279", f"{279 / 629:.3f}"]
    rates = build_report.baselines(_eval_rows())
    assert rows[3] == ["Class-matched random", "n/a", f"{rates['class_matched_random']:.3f}"]
    text = (site / "index.html").read_text(encoding="utf-8")
    assert "always saying down scores higher" in text
    assert "cost band, so a wrong call fixes" in text


def test_control_v3_section_refuses_data_missing_a_preregistered_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = (DATA_DIR / "detection_curve_v3.csv").read_text(encoding="utf-8").splitlines()
    kept = [lines[0]] + [line for line in lines[1:] if float(line.split(",")[0]) != 0.6]
    assert len(kept) < len(lines)
    (tmp_path / "detection_curve_v3.csv").write_text("\n".join(kept) + "\n", encoding="utf-8")
    monkeypatch.setattr(build_report, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="no rows"):
        build_report.control_v3_section()


def test_design_effect_table_has_one_row_per_effect(site: Path) -> None:
    rows = _tables(site)["design-effect"]
    assert len(rows) == len(build_report.DESIGN_EFFECTS)
    for row, effect in zip(rows, build_report.DESIGN_EFFECTS, strict=True):
        assert row[0] == f"{effect:g}"
        assert row[1] == f"{629 / effect:.1f}"
        assert row[3] in {"yes", "no"}


def test_by_recommendation_table_follows_the_csv(site: Path) -> None:
    rows = _tables(site)["by-recommendation"]
    by_rec = {r["recommendation"]: r for r in _eval_rows()}
    assert [row[0] for row in rows] == ["BUY", "SELL", "AVOID"]
    buy, sell, avoid = (by_rec[k] for k in ("BUY", "SELL", "AVOID"))
    up = int(buy["n_synth_correct"]) + sum(int(r["n_resolved"]) - int(r["n_synth_correct"]) for r in (sell, avoid))
    down = int(buy["n_resolved"]) - int(buy["n_synth_correct"]) + sum(int(r["n_synth_correct"]) for r in (sell, avoid))
    assert (up, down) == (279, 350)
    for row in rows:
        source = by_rec[row[0]]
        resolved, correct = int(source["n_resolved"]), int(source["n_synth_correct"])
        assert row[1] == f"{resolved:,}"
        assert row[2] == f"{correct:,}"
        assert row[3] == f"{correct / resolved:.3f}"
        matching = up if row[0] == "BUY" else down
        assert row[5] == f"{matching / 629:.3f}"
    assert [row[5] for row in rows] == ["0.444", "0.556", "0.556"]
    assert "not adjusted for multiplicity" in (site / "index.html").read_text(encoding="utf-8")


def test_control_v3_primary_rows_follow_the_csv_and_the_preregistered_code(site: Path) -> None:
    cells = _cells_v3()
    result = analyse(cells)
    signals = sorted({c.signal for c in cells})
    rows = _tables(site)["control-v3-primary"]
    assert [row[0] for row in rows] == [f"{s:.2f}" for s in signals]
    for row, signal in zip(rows, signals, strict=True):
        group = [c.model_balanced_accuracy for c in cells if c.signal == signal]
        values = [v for v in group if v is not None]
        endpoint = result["equivalence_zero_signal"] if signal == 0.0 else result["superiority"][signal]
        verdict = endpoint["verdict"]
        assert row[1] == str(len(values))
        assert row[2] == f"{statistics.fmean(values):.3f}"
        assert row[5] == _verdict_text(signal == 0.0, verdict)


def test_control_v3_states_the_seed_count_per_signal(site: Path) -> None:
    cells = _cells_v3()
    counts = {sum(1 for c in cells if c.signal == s) for s in {c.signal for c in cells}}
    assert len(counts) == 1
    text = (site / "index.html").read_text(encoding="utf-8")
    assert f"{counts.pop()} seeds per signal" in text
    assert "Pre-registered primary endpoints" in text
    assert len(_tables(site)["control-v3-secondary"]) == len({c.signal for c in cells})


def test_control_v3_rejects_an_unexpected_gate_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = "0.000000,1000,true,logistic,0.4,0.1,0.5,0.5,0.3,0.55,0.45,255"
    (tmp_path / "detection_curve_v3.csv").write_text(f"{V3_HEADER}\n{row}\n", encoding="utf-8")
    monkeypatch.setattr(build_report, "DATA_DIR", tmp_path)
    with pytest.raises(KeyError):
        build_report.control_v3_section()


def test_exploratory_section_reports_an_interval_that_contains_chance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _exploratory_csv(tmp_path, low=0.4950, high=0.5070)
    monkeypatch.setattr(build_report, "DATA_DIR", tmp_path)
    text = build_report.exploratory_section()
    assert "contains 0.50" in text
    assert "excludes 0.50" not in text
    assert "0.007 above 0.50" in text
    assert "does not identify the cause" in text
    assert "not pre-registered" in text


def test_exploratory_section_reports_an_interval_that_excludes_chance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _exploratory_csv(tmp_path, low=0.5020, high=0.5200)
    monkeypatch.setattr(build_report, "DATA_DIR", tmp_path)
    text = build_report.exploratory_section()
    assert "excludes 0.50" in text
    assert "contains 0.50" not in text
    assert "3 of 40 cells" in text


def test_exploratory_section_is_empty_when_the_file_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_report, "DATA_DIR", tmp_path)
    assert build_report.exploratory_section() == ""


def test_new_sections_keep_heading_levels_and_order(site: Path, page: _Page) -> None:
    text = (site / "index.html").read_text(encoding="utf-8")
    order = ["cohort", "evaluation-baselines", "control", "control-v3"]
    if 'id="exploratory-zero-signal"' in text:
        order.append("exploratory-zero-signal")
    order += ["control-seeds", "figures"]
    positions = [text.index(f'<section id="{name}"') for name in order]
    assert positions == sorted(positions)
    assert page.headings[0] == 1
    assert all(b - a <= 1 for a, b in zip(page.headings, page.headings[1:]))


def test_correction_paragraph_quotes_the_corrected_row(site: Path) -> None:
    """The literal cv_f1_mean in the 22 November 2026 correction paragraph equals the corrected data row."""
    with (DATA_DIR / "detection_curve_v3.csv").open(encoding="utf-8", newline="") as handle:
        row = next(r for r in csv.DictReader(handle) if float(r["signal"]) == 0.6 and r["seed"] == "1008")
    text = (site / "index.html").read_text(encoding="utf-8")
    assert f"from 0.665139 to {row['cv_f1_mean']}" in text
