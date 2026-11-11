from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "docs" / "img"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_figures", ROOT / "docs" / "report" / "build_figures.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_figures = _load()


def _build(out: Path) -> dict[str, bytes]:
    assert build_figures.main(["--out", str(out)]) == 0
    return {p.name: p.read_bytes() for p in sorted(out.glob("*.svg"))}


def test_build_is_byte_identical_across_runs(tmp_path: Path) -> None:
    first = _build(tmp_path / "a")
    second = _build(tmp_path / "b")
    assert first.keys() == second.keys()
    assert len(first) == len(build_figures.FIGURES)
    for name in first:
        assert first[name] == second[name], name


def _portable(svg: bytes) -> bytes:
    """Drop the two things that differ between platforms without changing the drawing: the sign of zero and the
    content-hashed element ids derived from it."""
    text = re.sub(rb"(?<![\d.])-0(?![\d.])", b"0", svg)
    return re.sub(rb"\b([mp])[0-9a-f]{10}\b", rb"\1ID", text)


def test_committed_figures_match_a_fresh_build(tmp_path: Path) -> None:
    fresh = _build(tmp_path / "fresh")
    committed = sorted(IMG_DIR.glob("*.svg"))
    assert [p.name for p in committed] == sorted(fresh)
    for path in committed:
        data = path.read_bytes()
        assert data, path.name
        stale = f"{path.name} is stale: rerun docs/report/build_figures.py"
        assert _portable(data) == _portable(fresh[path.name]), stale


def test_portable_form_ignores_only_zero_sign_and_hashed_ids() -> None:
    a = b'<path id="mc91ad6a70d" d="M -0 5.4 L -5.4 -0 L -0.5 1"/>'
    b = b'<path id="m236070abc9" d="M 0 5.4 L -5.4 0 L -0.5 1"/>'
    assert _portable(a) == _portable(b)
    assert _portable(a) != _portable(a.replace(b"5.4", b"5.5"))


def test_figures_hold_no_ticker_or_price_text() -> None:
    svgs = sorted(IMG_DIR.glob("*.svg"))
    assert svgs
    banned_words = re.compile(r"\b(profit|returns?|alpha|outperform\w*|performance)\b", re.IGNORECASE)
    for path in svgs:
        text = path.read_text(encoding="utf-8")
        for symbol in ("AAPL", "MSFT", "GOOGL", "AMZN", "META"):
            assert symbol not in text, (path.name, symbol)
        assert re.search(r"\$\s?\d", text) is None, path.name
        assert banned_words.search(text) is None, path.name


def test_wilson_interval_matches_hand_computed_values() -> None:
    # 5 of 10: centre 0.5, half-width 0.2634, so [0.2366, 0.7634].
    lo, hi = build_figures.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=5e-4)
    assert hi == pytest.approx(0.7634, abs=5e-4)
    # 0 of 4: lower bound 0, upper bound z^2 / (n + z^2) = 3.8415 / 7.8415 = 0.4899.
    lo, hi = build_figures.wilson(0, 4)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.4899, abs=5e-4)


def test_wilson_rejects_empty_sample() -> None:
    with pytest.raises(ValueError):
        build_figures.wilson(0, 0)


def test_sharpe_bins_are_numerically_ascending() -> None:
    order = build_figures.sharpe_bin_order()
    values = [float(label) for label in order]
    assert values == sorted(values)
    assert values == [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]


def test_ordered_bins_sorts_numerically_not_lexically() -> None:
    assert build_figures.ordered_bins(["-0.5", "-1.0", "-1.5", "0.0"]) == ["-1.5", "-1.0", "-0.5", "0.0"]
    assert build_figures.ordered_bins(["0.1-0.3", "-0.5--0.1"]) != []
    assert build_figures.ordered_bins(["b", "a"]) == ["b", "a"]
