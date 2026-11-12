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
DATA_DIR = ROOT / "docs" / "data"
T_975_19 = 2.093024


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
    assert f"two-sided p = {expected:.3f}" in (site / "index.html").read_text(encoding="utf-8")


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
