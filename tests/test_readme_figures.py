"""The README figures are drawn from the committed data and the code's own constants, and agree with the README text."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from aer.demo.control_v3 import _equivalence, _superiority
from aer.demo.prereg_v3 import ALPHA, DELTA, SIGNALS
from aer.ml.constants import CV_TEST_SIZE_BARS, MIN_TRAINING_BARS, N_CV_SPLITS, PURGE_GAP_BARS

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_figures", ROOT / "docs" / "report" / "build_figures.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass under `from __future__ import annotations` looks the class's
    # module up in sys.modules (Python docs recipe for importing a source file directly).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_figures = _load()


def test_rule_baselines_follow_from_the_recommendation_counts() -> None:
    assert build_figures.rule_baselines() == (314, 350, 279, 629)


def test_readme_quotes_the_counts_the_hero_figure_draws() -> None:
    correct, down, _, resolved = build_figures.rule_baselines()
    assert f"{correct} of {resolved}" in README
    assert f"{down} of {resolved}" in README


def test_v3_intervals_are_the_preregistered_intervals() -> None:
    points = {p.signal: p for p in build_figures.v3_points()}
    assert sorted(points) == sorted(SIGNALS)
    zero = _equivalence(list(points[0.0].values), DELTA, ALPHA)
    assert points[0.0].lower == pytest.approx(zero["lower"])
    assert points[0.0].upper == pytest.approx(zero["upper"])
    nonzero = {s: list(p.values) for s, p in points.items() if s > 0}
    superiority = _superiority(nonzero, ALPHA)
    for signal in nonzero:
        assert points[signal].lower == pytest.approx(superiority[signal]["lower_bound"])


def test_v3_figure_agrees_with_the_readme_table() -> None:
    points = {p.signal: p for p in build_figures.v3_points()}
    assert [round(points[s].mean, 3) for s in (0.0, 0.3, 0.6)] == [0.516, 0.530, 0.565]
    assert (round(points[0.0].lower, 3), round(points[0.0].upper, 3)) == (0.510, 0.522)
    assert [round(points[s].lower, 3) for s in (0.3, 0.6)] == [0.519, 0.547]
    assert [points[s].gate_passed for s in (0.0, 0.3, 0.6)] == [4, 5, 11]
    assert {p.seeds for p in points.values()} == {46}


def test_cv_geometry_is_the_code_layout() -> None:
    folds, cv_block, holdout = build_figures.cv_layout(MIN_TRAINING_BARS)
    assert len(folds) == N_CV_SPLITS
    for train, valid in folds:
        assert len(valid) == CV_TEST_SIZE_BARS
        assert valid.start - train.stop == PURGE_GAP_BARS
        assert train.start == 0
    assert folds[-1][1].stop == cv_block.stop
    assert holdout.start - cv_block.stop == PURGE_GAP_BARS
    assert holdout.stop == MIN_TRAINING_BARS


def test_new_figures_are_in_the_build() -> None:
    names = {name for name, _ in build_figures.FIGURES}
    assert {"hero_evidence.svg", "detection_curve_v3.svg", "cv_geometry.svg"} <= names


def test_every_image_the_readme_embeds_is_committed() -> None:
    paths = re.findall(r'<img src="([^"]+)"', README)
    assert paths
    for path in paths:
        assert (ROOT / path).is_file(), path


def test_readme_flowcharts_use_flowchart_shapes() -> None:
    blocks = re.findall(r"```mermaid\n(.*?)```", README, flags=re.DOTALL)
    assert len(blocks) >= 4
    for block in blocks:
        assert "flowchart" in block
        assert "classDef" in block
        assert "([" in block, "each flowchart starts or ends at a terminator"


# Mermaid flowchart keywords that fail as node ids on GitHub ("call" broke the routing chart on 17 Nov 2026).
MERMAID_KEYWORDS = {"call", "click", "class", "classDef", "default", "end", "graph", "href", "style", "subgraph"}
EDGE = r"(?:-->|-\.->)\s*(?:\|[^|]*\|\s*)?"
NODE_WITH_SHAPE = re.compile(rf"(?:^\s*|{EDGE})(\w+)\s*[\[\(\{{]", re.M)


def test_readme_flowcharts_use_no_keyword_as_a_node_id() -> None:
    blocks = re.findall(r"```mermaid\n(.*?)```", README, flags=re.DOTALL)
    for block in blocks:
        ids = set(NODE_WITH_SHAPE.findall(block))
        ids |= {i for line in re.findall(r"^\s*class (\S+) \w+$", block, re.M) for i in line.split(",")}
        assert not ids & MERMAID_KEYWORDS, sorted(ids & MERMAID_KEYWORDS)
