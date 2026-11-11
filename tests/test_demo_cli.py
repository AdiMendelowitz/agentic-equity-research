import re
from pathlib import Path

import pytest

from aer.demo.banner import render_banner
from aer.demo.cli import main, render_brief

ROOT = Path(__file__).resolve().parents[1]


def _footprint() -> tuple[list[str], list[str]]:
    top = sorted(p.name for p in ROOT.iterdir())
    models = sorted(
        str(p.relative_to(ROOT))
        for folder in (ROOT / "src", ROOT / "tests")
        for p in folder.rglob("*")
        if p.suffix == ".joblib" or "ml_models" in p.parts
    )
    return top, models


def test_synthetic_banner_states_no_real_data():
    text = render_banner()
    assert text.startswith("SYNTHETIC DATA")
    assert "No market data and no real company." in text
    assert "Not investment advice" in text


@pytest.mark.integration
def test_synthetic_run_is_offline_and_writes_nothing(capsys):
    before = _footprint()
    assert main(["--tickers", "2", "--days", "300"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("SYNTHETIC DATA")
    assert out.count("Brief: SYN") == 2
    assert "Tickers: 2" in out
    assert _footprint() == before


def test_brief_numbers_come_from_the_values():
    values: dict[str, object] = {
        "ticker": "SYN1",
        "sessions": 731,
        "close": 123.4567,
        "rsi": 61.2345,
        "sma50": 98.7654,
        "trend": "RISING",
        "gate": "predicted",
        "probability": 0.5678,
        "reason": None,
        "rows": None,
        "min_rows": None,
        "cv_f1": None,
    }
    text = render_brief(values)
    allowed = {f"{v:.2f}" if isinstance(v, float) else str(v) for v in values.values() if isinstance(v, (int, float))}
    numbers = re.findall(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?", text)
    assert numbers
    assert set(numbers) <= allowed
    assert {"123.46", "61.23", "98.77", "0.57", "731"} <= set(numbers)


def test_abstained_brief_reports_the_reason():
    values: dict[str, object] = {
        "ticker": "SYN2",
        "sessions": 300,
        "close": 100.0,
        "rsi": None,
        "sma50": 99.5,
        "trend": "FLAT",
        "gate": "abstained",
        "reason": "too few labelled rows for cross-validation",
        "rows": 211,
        "min_rows": 400,
        "cv_f1": None,
    }
    text = render_brief(values)
    assert "abstained" in text
    assert "RSI: n/a" in text
    allowed = {"300", "100.00", "99.50", "211", "400"}
    assert set(re.findall(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?", text)) <= allowed
