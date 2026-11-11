"""Leakage guards for the ML feature matrix: label geometry, causality of features and CV purge gap."""

import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pandas.testing import assert_frame_equal
from sklearn.model_selection import TimeSeriesSplit

import aer.ml.models as models_module
from aer.ml.constants import (
    CV_TEST_SIZE_BARS,
    FEATURE_COLS,
    MIN_TRAINING_BARS,
    N_CV_SPLITS,
    PREDICTION_HORIZON_DAYS,
    PURGE_GAP_BARS,
    ROUND_TRIP_COST_BPS,
)
from aer.ml.features import build_feature_matrix
from aer.utils.trading_calendar import add_trading_days

N_BARS = 700
PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _synthetic_frame() -> pd.DataFrame:
    sessions = mcal.get_calendar("NYSE").valid_days(start_date="2022-01-03", end_date="2025-12-31")
    idx = pd.DatetimeIndex(sessions).tz_localize(None)[:N_BARS]
    rng = np.random.default_rng(1234)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, N_BARS))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, N_BARS)),
            "High": close * (1 + rng.uniform(0.001, 0.02, N_BARS)),
            "Low": close * (1 - rng.uniform(0.001, 0.02, N_BARS)),
            "Close": close,
            "Volume": rng.integers(500_000, 5_000_000, N_BARS).astype(float),
            "Stock Splits": np.zeros(N_BARS),
        },
        index=idx,
    )


def _matrix(frame: pd.DataFrame) -> pd.DataFrame:
    with patch("aer.ml.features.get_price_history_raw", return_value=frame):
        return build_feature_matrix("SYNTH")


def test_label_is_forward_return_at_horizon() -> None:
    frame = _synthetic_frame()
    matrix = _matrix(frame)
    close = pd.Series(frame["Close"].values, index=[d.date() for d in frame.index])
    threshold = ROUND_TRIP_COST_BPS / 10_000

    labelled = matrix["label"].dropna()
    assert len(labelled) > 0
    for ts, label in labelled.items():
        row_date = ts.date()
        target = add_trading_days(row_date, PREDICTION_HORIZON_DAYS)
        expected = float(close[target] / close[row_date] - 1 > threshold)
        assert label == expected, f"label mismatch on {row_date}"

    last_dates = set(frame.index[-PREDICTION_HORIZON_DAYS:])
    tail_labels = matrix.loc[matrix.index.isin(last_dates), "label"]
    assert tail_labels.isna().all()


def test_features_do_not_depend_on_future_bars() -> None:
    frame = _synthetic_frame()
    base = _matrix(frame)

    cut_pos = int(N_BARS * 0.7)
    cut = frame.index[cut_pos]
    rng = np.random.default_rng(99)
    perturbed = frame.copy()  # factors stay within the integrity validator's 50% daily-jump limit
    n_future = N_BARS - cut_pos - 1
    for col in PRICE_COLS:
        perturbed.iloc[cut_pos + 1 :, perturbed.columns.get_loc(col)] *= rng.uniform(0.85, 1.15, n_future)
    other = _matrix(perturbed)

    common = base.index.intersection(other.index)
    common_past = common[common <= cut]
    assert len(common_past) > 100
    feature_cols = [c for c in FEATURE_COLS if c in base.columns]
    assert_frame_equal(
        base.loc[common_past, feature_cols], other.loc[common_past, feature_cols], check_exact=False, atol=1e-12
    )

    cut_date = cut.date()
    resolved = [d for d in common_past if add_trading_days(d.date(), PREDICTION_HORIZON_DAYS) <= cut_date]
    assert len(resolved) > 100
    assert_frame_equal(base.loc[resolved, ["label"]], other.loc[resolved, ["label"]])


def _timeseries_split_keywords(source: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name == "TimeSeriesSplit":
                found.append({kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg})
    return found


def test_purge_gap_geometry() -> None:
    assert PURGE_GAP_BARS == 10
    assert PURGE_GAP_BARS >= 2 * PREDICTION_HORIZON_DAYS

    splitter = TimeSeriesSplit(n_splits=N_CV_SPLITS, gap=PURGE_GAP_BARS, test_size=CV_TEST_SIZE_BARS)
    folds = list(splitter.split(np.arange(MIN_TRAINING_BARS)))
    assert len(folds) == N_CV_SPLITS
    for train, test in folds:
        assert train.max() < test.min()
        assert test.min() - train.max() - 1 >= PURGE_GAP_BARS

    source = Path(models_module.__file__).read_text(encoding="utf-8")
    calls = _timeseries_split_keywords(source)
    assert calls, "no TimeSeriesSplit call found in models.py"
    assert any(
        kw.get("gap") == "PURGE_GAP_BARS"
        and kw.get("n_splits") == "N_CV_SPLITS"
        and kw.get("test_size") == "CV_TEST_SIZE_BARS"
        for kw in calls
    )
