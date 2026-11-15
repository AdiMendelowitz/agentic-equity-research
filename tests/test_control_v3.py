from __future__ import annotations

import re
from pathlib import Path

import pytest
from scipy.stats import t

from aer.demo import prereg_v3
from aer.demo.control_v3 import CSV_HEADER_V3, CellV3, analyse, holm_adjust, run_cell_v3, write_csv_v3

ROOT = Path(__file__).resolve().parent.parent
SPEC_MD = ROOT / "docs" / "prereg" / "control-v3.md"

NEAR_HALF: list[float | None] = [0.495, 0.505, 0.500, 0.498, 0.502, 0.501, 0.499, 0.503]
LOW: list[float | None] = [0.45, 0.47, 0.46, 0.455, 0.465, 0.46, 0.46, 0.46]
CLEAR_MID: list[float | None] = [0.56, 0.58, 0.57, 0.59, 0.55, 0.57, 0.58, 0.56]
CLEAR_HIGH: list[float | None] = [0.69, 0.71, 0.70, 0.72, 0.68, 0.70, 0.71, 0.69]


def _cells(
    signal: float,
    model: list[float | None],
    oracle: list[float | None] | float | None = 0.6,
    passed: list[bool] | None = None,
) -> list[CellV3]:
    cells: list[CellV3] = []
    for i, ba in enumerate(model):
        o = oracle[i] if isinstance(oracle, list) else oracle
        gate = passed[i] if passed is not None else True
        used = "lightgbm" if ba is not None else None
        cells.append(CellV3(signal, 1000 + i, gate, used, 0.55, 0.02, ba, o, 0.24, 0.52, 0.48, 250))
    return cells


def _grid(
    zero: list[float | None] = NEAR_HALF,
    mid: list[float | None] = CLEAR_MID,
    high: list[float | None] = CLEAR_HIGH,
) -> list[CellV3]:
    return _cells(0.0, zero, 0.5) + _cells(0.3, mid, 0.64) + _cells(0.6, high, 0.72)


def test_holm_hand_values() -> None:
    assert holm_adjust([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert holm_adjust([0.04, 0.01]) == pytest.approx([0.04, 0.02])
    assert holm_adjust([0.30, 0.20]) == pytest.approx([0.40, 0.40])
    assert holm_adjust([0.6, 0.7]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([0.01, 0.02, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert holm_adjust([]) == []


def test_equivalence_true_when_centred_on_chance() -> None:
    eq = analyse(_grid())["equivalence_zero_signal"]
    assert eq["verdict"] is True
    assert eq["n"] == 8
    assert eq["lower"] > 0.47 and eq["upper"] < 0.53
    assert eq["interval_confidence"] == pytest.approx(0.9)


def test_equivalence_false_when_centred_on_046() -> None:
    eq = analyse(_grid(zero=LOW))["equivalence_zero_signal"]
    assert eq["verdict"] is False
    assert eq["mean"] == pytest.approx(0.46, abs=1e-3)


def test_equivalence_interval_matches_hand_formula() -> None:
    values = [0.49, 0.51, 0.50, 0.52]
    eq = analyse(_cells(0.0, list(values), 0.5))["equivalence_zero_signal"]
    mean = sum(values) / 4
    sd = (sum((v - mean) ** 2 for v in values) / 3) ** 0.5
    half = float(t.ppf(0.95, 3)) * sd / 2
    assert eq["lower"] == pytest.approx(mean - half)
    assert eq["upper"] == pytest.approx(mean + half)


def test_superiority_true_when_clearly_above_chance() -> None:
    result = analyse(_grid())["superiority"]
    assert set(result) == {0.3, 0.6}
    assert result[0.3]["verdict"] is True
    assert result[0.6]["verdict"] is True
    assert result[0.3]["lower_bound"] > 0.5
    assert result[0.3]["p_holm"] >= result[0.3]["p_raw"]


def test_superiority_false_when_at_chance() -> None:
    result = analyse(_grid(mid=NEAR_HALF))["superiority"]
    assert result[0.3]["verdict"] is False
    assert result[0.6]["verdict"] is True


def test_superiority_holm_uses_all_non_zero_signals() -> None:
    result = analyse(_grid())["superiority"]
    raw = sorted([result[0.3]["p_raw"], result[0.6]["p_raw"]])
    expected = holm_adjust(raw)
    assert sorted([result[0.3]["p_holm"], result[0.6]["p_holm"]]) == pytest.approx(sorted(expected))


def test_paired_gap_equals_hand_mean_and_interval() -> None:
    cells = _cells(0.0, [0.5, 0.5, 0.5], 0.5) + _cells(0.3, [0.5, 0.6, 0.7], [0.6, 0.8, 0.7])
    gap = analyse(cells)["paired_gap"][0.3]
    assert gap["n"] == 3
    assert gap["mean"] == pytest.approx(0.1)
    half = float(t.ppf(0.975, 2)) * 0.1 / 3**0.5
    assert gap["lower"] == pytest.approx(0.1 - half)
    assert gap["upper"] == pytest.approx(0.1 + half)


def test_none_cells_are_excluded_and_n_reported() -> None:
    zero = list(NEAR_HALF)
    zero[0] = None
    zero[3] = None
    mid = list(CLEAR_MID)
    mid[1] = None
    result = analyse(_grid(zero=zero, mid=mid))
    assert result["equivalence_zero_signal"]["n"] == 6
    assert result["superiority"][0.3]["n"] == 7
    assert result["superiority"][0.6]["n"] == 8
    assert result["paired_gap"][0.3]["n"] == 7
    assert result["paired_gap"][0.0]["n"] == 6
    assert result["gate"][0.0] == {"n_cells": 8, "n_fitted": 6, "gate_passed": 8}


def test_too_few_values_give_no_verdict() -> None:
    result = analyse(_cells(0.0, [0.5, None, None], 0.5) + _cells(0.3, [0.6, None, None], 0.64))
    assert result["equivalence_zero_signal"]["verdict"] is None
    assert result["superiority"][0.3]["verdict"] is None
    assert result["paired_gap"][0.3]["mean"] is None


def test_gate_counts_per_signal() -> None:
    cells = _cells(0.0, [0.5] * 4, 0.5, [True, False, False, False]) + _cells(0.3, [0.6] * 4, 0.64, [True] * 4)
    gate = analyse(cells)["gate"]
    assert gate[0.0]["gate_passed"] == 1
    assert gate[0.3]["gate_passed"] == 4


def test_write_csv_v3_header_and_rows(tmp_path: Path) -> None:
    cells = [
        CellV3(0.3, 1001, True, "lightgbm", 0.5312345678, 0.0123, 0.55, 0.7, 0.1912, 0.52, 0.48, 99),
        CellV3(0.0, 1000, False, None, None, None, None, None, None, 0.5, 0.5, 0),
    ]
    path = tmp_path / "out" / "curve_v3.csv"
    write_csv_v3(cells, path)
    raw = path.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").split("\n")
    assert lines[0] == CSV_HEADER_V3
    assert lines[0].split(",")[:4] == ["signal", "seed", "gate_passed", "model_used"]
    assert lines[1] == "0.300000,1001,True,lightgbm,0.531235,0.012300,0.550000,0.700000,0.191200,0.520000,0.480000,99"
    assert lines[2] == "0.000000,1000,False" + "," * 7 + "0.500000,0.500000,0"
    assert lines[3] == ""
    assert len(lines[1].split(",")) == len(CSV_HEADER_V3.split(","))


def test_seeds_raise_when_unset_and_are_disjoint_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prereg_v3, "N_SEEDS", None)
    with pytest.raises(ValueError):
        prereg_v3.seeds()
    monkeypatch.setattr(prereg_v3, "N_SEEDS", 25)
    found = prereg_v3.seeds()
    assert found == tuple(range(prereg_v3.SEED_OFFSET, prereg_v3.SEED_OFFSET + 25))
    assert set(found).isdisjoint(range(20))


def _constant(text: str, name: str) -> str:
    match = re.search(rf"^- {name} = (.+?)\s*$", text, flags=re.MULTILINE)
    assert match is not None, f"{name} missing from the specification"
    return match.group(1)


def test_markdown_matches_prereg_constants() -> None:
    text = SPEC_MD.read_text(encoding="utf-8")
    assert float(_constant(text, "DELTA")) == prereg_v3.DELTA
    assert float(_constant(text, "ALPHA")) == prereg_v3.ALPHA
    assert float(_constant(text, "POWER")) == prereg_v3.POWER
    assert float(_constant(text, "SD_UPPER_CONFIDENCE")) == prereg_v3.SD_UPPER_CONFIDENCE
    assert int(_constant(text, "MIN_SEEDS")) == prereg_v3.MIN_SEEDS
    assert int(_constant(text, "SEED_OFFSET")) == prereg_v3.SEED_OFFSET
    signals = tuple(float(s) for s in _constant(text, "SIGNALS").split(","))
    assert signals == prereg_v3.SIGNALS
    assert prereg_v3.SPEC_ID in text


def test_markdown_states_n_seeds_consistently() -> None:
    text = SPEC_MD.read_text(encoding="utf-8")
    match = re.search(r"(?<![A-Z_])N_SEEDS = (\d+|to be set)", text)
    assert match is not None
    if prereg_v3.N_SEEDS is None:
        assert match.group(1) == "to be set"
    else:
        assert match.group(1) == str(prereg_v3.N_SEEDS)


@pytest.mark.integration
def test_run_cell_v3_end_to_end() -> None:
    cell = run_cell_v3(0.6, prereg_v3.SEED_OFFSET)
    assert cell.n_oos > 0
    assert cell.model_balanced_accuracy is not None and 0.0 <= cell.model_balanced_accuracy <= 1.0
    assert cell.oracle_balanced_accuracy is not None and 0.0 <= cell.oracle_balanced_accuracy <= 1.0
    assert cell.model_used in {"lightgbm", "logistic"}
    assert 0.5 <= cell.majority_accuracy <= 1.0
