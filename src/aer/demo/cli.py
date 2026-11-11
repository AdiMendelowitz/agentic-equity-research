"""Console entry point `aer-demo`: an offline run on synthetic data."""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import aer.ml.constants as ml_constants
import aer.ml.models as ml_models
from aer.data.integrity import MIN_PRICE_ROWS
from aer.demo.banner import render_banner
from aer.demo.generate import SyntheticSeries, synthetic_universe
from aer.indicators.technical import compute_indicators
from aer.ml.features import build_feature_matrix
from aer.utils.python_replacements import compute_trend


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_brief(values: dict[str, object]) -> str:
    lines = [
        f"Brief: {values['ticker']}",
        f"  Sessions: {_fmt(values['sessions'])}",
        f"  Close: {_fmt(values['close'])}",
        f"  RSI: {_fmt(values['rsi'])}",
        f"  SMA50: {_fmt(values['sma50'])}",
        f"  Recent trend: {values['trend']}",
    ]
    if values["gate"] == "predicted":
        lines.append(f"  ML gate: predicted, calibrated probability of an up move {_fmt(values['probability'])}")
    else:
        lines.append(f"  ML gate: abstained, {values['reason']}")
        if values.get("rows") is not None:
            lines.append(f"  Rows available: {_fmt(values['rows'])}, required: {_fmt(values['min_rows'])}")
        if values.get("cv_f1") is not None:
            lines.append(f"  Cross-validated macro F1: {_fmt(values['cv_f1'])}")
    return "\n".join(lines)


def _abstained(reason: str) -> dict[str, object]:
    return {"gate": "abstained", "reason": reason, "probability": None, "rows": None, "min_rows": None, "cv_f1": None}


def _gate_values(series: SyntheticSeries, model_dir: Path) -> dict[str, object]:
    if len(series.prices) < MIN_PRICE_ROWS:
        out = _abstained("too few sessions for the feature builder")
        out.update(rows=len(series.prices), min_rows=MIN_PRICE_ROWS)
        return out
    saved = (ml_constants.MODEL_DIR, ml_models.MODEL_DIR)
    ml_constants.MODEL_DIR = model_dir
    ml_models.MODEL_DIR = model_dir
    try:
        matrix = build_feature_matrix(series.ticker, prices=series.prices)
        result = ml_models.train_and_predict(matrix, series.ticker)
    except Exception as exc:
        return _abstained(f"exception {type(exc).__name__}")
    finally:
        ml_constants.MODEL_DIR, ml_models.MODEL_DIR = saved
    if result is None:
        return _abstained("no result from the gate")
    prediction = result[0]
    if not prediction.prediction_skipped:
        probability = prediction.calibrated_prob_up
        if probability is None:
            probability = prediction.prob_up_raw
        return {"gate": "predicted", "probability": probability}
    rows = prediction.trained_on_bars
    if rows is not None and rows < ml_constants.MIN_TRAINING_BARS:
        out = _abstained("too few labelled rows for cross-validation")
        out.update(rows=rows, min_rows=ml_constants.MIN_TRAINING_BARS)
        return out
    out = _abstained("cross-validation score below the gate threshold")
    out["cv_f1"] = prediction.cv_f1_macro
    return out


def _brief_values(series: SyntheticSeries, model_dir: Path) -> dict[str, object]:
    indicators = compute_indicators(series.prices.copy())
    values: dict[str, object] = {
        "ticker": series.ticker,
        "sessions": len(series.prices),
        "close": indicators["price"],
        "rsi": indicators["rsi"],
        "sma50": indicators["sma50"],
        "trend": compute_trend(series.prices["Close"], n=20),
    }
    values.update(_gate_values(series, model_dir))
    return values


def _run_synthetic(args: argparse.Namespace) -> int:
    started = time.monotonic()
    universe = synthetic_universe(n_tickers=args.tickers, n_sessions=args.days, seed=args.seed, signal=args.signal)
    predicted = 0
    briefs: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "ml_models"
        model_dir.mkdir()
        for series in universe:
            values = _brief_values(series, model_dir)
            if values["gate"] == "predicted":
                predicted += 1
            briefs.append(render_brief(values))
    print(render_banner("synthetic"))
    print()
    print("\n\n".join(briefs))
    print()
    print(f"Tickers: {len(universe)}, predicted: {predicted}, elapsed seconds: {time.monotonic() - started:.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aer-demo", description="Offline demonstration of the research pipeline.")
    parser.add_argument("--tickers", type=int, default=5, choices=range(1, 6), help="synthetic tickers (default 5)")
    parser.add_argument("--days", type=int, default=700, help="synthetic sessions per ticker (default 700)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    parser.add_argument("--signal", type=float, default=0.0, help="planted signal strength (default 0.0)")
    args = parser.parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        # keep console output UTF-8 on Windows code pages
        reconfigure(encoding="utf-8", errors="replace")
    return _run_synthetic(args)
