from __future__ import annotations

from pathlib import Path

import pytest

import aer.ml.models as ml_models
from aer.demo.control import (
    CSV_HEADER,
    ControlCell,
    _skip_reason,
    analytic_ceiling,
    balanced_accuracy,
    evaluate_expectations,
    run_cell,
    summarise,
    write_csv,
)

ROOT = Path(__file__).resolve().parent.parent

NEAR_HALF: list[float | None] = [0.48, 0.52, 0.50, 0.51, 0.49]
MID: list[float | None] = [0.58, 0.60, 0.62, 0.59, 0.61]
HIGH: list[float | None] = [0.68, 0.70, 0.72, 0.69, 0.71]
VERY_HIGH: list[float | None] = [0.88, 0.90, 0.92, 0.89, 0.91]
ABSENT: list[float | None] = [None] * 5


def _cells(signal: float, model: list[float | None], oracle: float | None, f1: float | None = 0.5) -> list[ControlCell]:
    return [
        ControlCell(signal, seed, True, "predicted", f1, 0.02, ba, oracle, 0.52, 0.48, 99)
        for seed, ba in enumerate(model)
    ]


def _grid(
    zero: list[float | None] = NEAR_HALF,
    mid: list[float | None] = MID,
    high: list[float | None] = HIGH,
    oracle_offset: float = 0.0,
    oracle_zero: float = 0.5,
) -> list[ControlCell]:
    return (
        _cells(0.0, zero, oracle_zero)
        + _cells(0.3, mid, analytic_ceiling(0.3) + oracle_offset)
        + _cells(0.6, high, analytic_ceiling(0.6) + oracle_offset)
    )


def test_write_csv_header_and_formatting(tmp_path: Path) -> None:
    cells = [
        ControlCell(0.3, 2, True, "predicted", 0.5312345678, 0.0123, 0.55, 0.7, 0.52, 0.48, 99),
        ControlCell(0.0, 0, False, "too_few_bars", None, None, None, None, 0.5, 0.5, 0),
    ]
    path = tmp_path / "out" / "curve.csv"
    write_csv(cells, path)
    raw = path.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").split("\n")
    expected = (
        "signal,seed,gate_predicted,gate_reason,cv_f1_mean,cv_f1_std,model_balanced_accuracy,"
        "oracle_balanced_accuracy,majority_accuracy,label_base_rate,n_oos"
    )
    assert lines[0] == expected
    assert lines[0] == CSV_HEADER
    assert lines[1] == "0.300000,2,True,predicted,0.531235,0.012300,0.550000,0.700000,0.520000,0.480000,99"
    assert lines[2] == "0.000000,0,False,too_few_bars,,,,,0.500000,0.500000,0"
    assert lines[3] == ""


def test_balanced_accuracy_hand_cases() -> None:
    assert balanced_accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert balanced_accuracy([1, 0, 1, 0], [0, 0, 0, 0]) == 0.5
    assert balanced_accuracy([1, 1, 1], [1, 0, 1]) is None
    assert balanced_accuracy([0, 0], [0, 1]) is None
    # truth [1,1,1,0,0], prediction [1,1,0,0,1]: recall on 1 is 2/3, recall on 0 is 1/2, mean 7/12
    assert balanced_accuracy([1, 1, 1, 0, 0], [1, 1, 0, 0, 1]) == pytest.approx(7 / 12)


def test_analytic_ceiling_values() -> None:
    assert analytic_ceiling(0.0) == 0.5
    assert analytic_ceiling(0.3) == pytest.approx(0.6413, abs=1e-3)
    assert analytic_ceiling(0.6) == pytest.approx(0.7167, abs=1e-3)


def test_skip_reason_branches() -> None:
    assert _skip_reason(None, None) == "too_few_bars"
    assert _skip_reason(0.50, 0.05) == "f1_mean_below_minimum"
    assert _skip_reason(0.60, 0.20) == "f1_std_above_limit"
    assert _skip_reason(0.50, 0.20) == "f1_mean_below_minimum_and_std_above_limit"


def test_e0_met_and_not_met() -> None:
    assert evaluate_expectations(_grid())["V2-E0"] is True
    assert evaluate_expectations(_grid(oracle_offset=0.1))["V2-E0"] is False
    assert evaluate_expectations(_grid(oracle_zero=0.55))["V2-E0"] is False


def test_e1_met_not_met_and_none() -> None:
    assert evaluate_expectations(_grid())["V2-E1"] is True
    assert evaluate_expectations(_grid(zero=MID))["V2-E1"] is False
    assert evaluate_expectations(_grid(zero=ABSENT))["V2-E1"] is None


def test_e2_met_and_not_met() -> None:
    assert evaluate_expectations(_grid())["V2-E2"] is True
    assert evaluate_expectations(_grid(high=NEAR_HALF))["V2-E2"] is False


def test_e3_met_not_met_and_none() -> None:
    assert evaluate_expectations(_grid())["V2-E3"] is True
    assert evaluate_expectations(_grid(mid=VERY_HIGH))["V2-E3"] is False
    assert evaluate_expectations(_grid(mid=ABSENT))["V2-E3"] is None


def test_e4_met_and_not_met() -> None:
    assert evaluate_expectations(_grid())["V2-E4"] is True
    assert evaluate_expectations(_grid(high=VERY_HIGH))["V2-E4"] is False


def test_summarise_reports_one_row_per_signal() -> None:
    rows = summarise(_grid())
    assert [r["signal"] for r in rows] == [0.0, 0.3, 0.6]
    assert all(r["n"] == 5 and r["gate_predicted_share"] == 1.0 for r in rows)


@pytest.mark.integration
def test_run_cell_end_to_end_stays_inside_tmp_path(tmp_path: Path) -> None:
    watched = [ROOT / "ml_models", ROOT / "artifacts"]

    def snapshot() -> set[Path]:
        return {p for d in watched if d.exists() for p in d.rglob("*")}

    before = snapshot()
    model_dir_before = ml_models.MODEL_DIR
    cell = run_cell(0.6, 0, tmp_path / "control")
    assert isinstance(cell, ControlCell)
    assert isinstance(cell.gate_predicted, bool)
    assert cell.n_oos > 0
    assert cell.model_balanced_accuracy is not None and 0.0 <= cell.model_balanced_accuracy <= 1.0
    assert cell.oracle_balanced_accuracy is not None and 0.0 <= cell.oracle_balanced_accuracy <= 1.0
    assert 0.5 <= cell.majority_accuracy <= 1.0
    assert 0.0 <= cell.label_base_rate <= 1.0
    assert cell.gate_reason in {"predicted", "too_few_bars", "f1_mean_below_minimum", "f1_std_above_limit"} or (
        cell.gate_reason == "f1_mean_below_minimum_and_std_above_limit"
    )
    assert ml_models.MODEL_DIR == model_dir_before
    assert snapshot() == before
