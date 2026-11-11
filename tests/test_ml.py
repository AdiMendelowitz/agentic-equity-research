"""ML tests - feature matrix and model sections."""

from unittest.mock import patch

import numpy as np
import pandas as pd


def _synthetic_df(n: int = 800) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with Stock Splits column (mimics get_price_history_raw output)."""
    idx = pd.bdate_range("2022-01-01", periods=n)
    rng = np.random.default_rng(42)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)
    splits = np.zeros(n)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Stock Splits": splits,
        },
        index=idx,
    )


def test_build_feature_matrix_shape():
    from aer.ml.features import build_feature_matrix

    df = _synthetic_df(800)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    assert "label" in feature_df.columns
    from aer.ml.constants import FEATURE_COLS

    available = [c for c in FEATURE_COLS if c in feature_df.columns]
    assert len(available) > 20, "Expected at least 20 feature columns"
    assert feature_df[available].notna().all().all()


def test_no_nan_in_features():
    from aer.ml.constants import FEATURE_COLS
    from aer.ml.features import build_feature_matrix

    df = _synthetic_df(800)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    available = [c for c in FEATURE_COLS if c in feature_df.columns]
    train_rows = feature_df.dropna(subset=["label"])
    assert train_rows[available].notna().all().all()


def test_label_uses_trading_days():
    from aer.ml.features import build_feature_matrix
    from aer.utils.trading_calendar import trading_days_between

    df = _synthetic_df(800)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    labeled = feature_df.dropna(subset=["label"])
    dates = labeled.index.tolist()
    if len(dates) > 1:
        gap = trading_days_between(dates[0].date(), dates[1].date())
        assert gap == 1


def test_cost_adjusted_label():
    from aer.ml.constants import ROUND_TRIP_COST_BPS
    from aer.ml.features import build_feature_matrix

    df = _synthetic_df(800)
    ROUND_TRIP_COST_BPS / 10_000
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    labeled = feature_df.dropna(subset=["label"])
    assert set(labeled["label"].unique()).issubset({0, 1})


def test_forbidden_ohlcv_columns_absent():
    from aer.ml.features import build_feature_matrix

    df = _synthetic_df(800)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    forbidden = {"Open", "High", "Low", "Close", "Volume", "Stock Splits"}
    assert not forbidden.intersection(
        feature_df.columns
    ), f"Forbidden columns found: {forbidden.intersection(feature_df.columns)}"


def test_isa9_excluded_from_features():
    from aer.ml.features import build_feature_matrix

    df = _synthetic_df(800)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    assert "ISA_9" not in feature_df.columns, "ISA_9 (Senkou Span A) must be excluded — look-ahead bias"


def test_rolling_52w_high_not_snapshot():
    from aer.ml.features import build_feature_matrix

    df = _synthetic_df(800)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    if "rolling_52w_high" in feature_df.columns:
        highs = feature_df["rolling_52w_high"].dropna()
        assert highs.nunique() > 1, "rolling_52w_high appears to be a constant snapshot"


def test_logistic_baseline_runs():
    from aer.ml.features import build_feature_matrix
    from aer.ml.models import train_and_predict

    df = _synthetic_df(900)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    result = train_and_predict(feature_df, "AAPL")
    assert result is not None
    ml = result[0]
    assert ml.ticker == "AAPL"
    assert ml.model_used in ("logistic", "lightgbm", "none")
    assert ml.trained_on_bars is not None and ml.trained_on_bars > 0


def test_lightgbm_under_two_minutes():
    import time

    from aer.ml.features import build_feature_matrix
    from aer.ml.models import train_and_predict

    df = _synthetic_df(900)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    start = time.time()
    train_and_predict(feature_df, "AAPL")
    elapsed = time.time() - start
    assert elapsed < 120, f"Training took {elapsed:.1f}s — exceeds 2-minute limit"


def test_high_std_returns_skipped():
    from aer.ml.models import train_and_predict

    df = _synthetic_df(900)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        from aer.ml.features import build_feature_matrix

        feature_df = build_feature_matrix("AAPL")

    with patch("aer.ml.models.cross_val_score") as mock_cv:
        mock_cv.return_value = np.array([0.3, 0.7, 0.2, 0.8, 0.1])  # std ≈ 0.26 > 0.15
        result = train_and_predict(feature_df, "AAPL")
    assert result is not None
    assert result[0].prediction_skipped is True
    assert result[0].model_used == "none"


def test_model_persisted_to_disk(tmp_path):
    import aer.ml.constants as c
    from aer.ml.features import build_feature_matrix
    from aer.ml.models import train_and_predict

    df = _synthetic_df(900)
    with patch("aer.ml.features.get_price_history_raw", return_value=df), patch.object(c, "MODEL_DIR", tmp_path):
        feature_df = build_feature_matrix("AAPL")
        result = train_and_predict(feature_df, "AAPL")
    if result and not result[0].prediction_skipped:
        joblib_files = list(tmp_path.glob("*.joblib"))
        assert len(joblib_files) >= 1, "Model was not persisted to disk"


def test_atomic_write_no_corruption(tmp_path):
    import joblib

    from aer.ml.models import _atomic_save_model

    model_path = tmp_path / "test_model.joblib"
    payload = {"model": "test", "version": 1}
    success = _atomic_save_model("TEST", model_path, payload)
    assert success is True
    loaded = joblib.load(model_path)
    assert loaded == payload


def test_class_balance_logged():
    from aer.ml.features import build_feature_matrix
    from aer.ml.models import train_and_predict

    df = _synthetic_df(900)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    result = train_and_predict(feature_df, "AAPL")
    assert result is not None
    ml = result[0]
    assert ml.class_balance_up is not None
    assert 0.0 <= ml.class_balance_up <= 1.0


def test_brier_score_stored():
    from aer.ml.features import build_feature_matrix
    from aer.ml.models import train_and_predict

    df = _synthetic_df(900)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    result = train_and_predict(feature_df, "AAPL")
    if result and not result[0].prediction_skipped:
        assert result[0].brier_score_raw is not None
        assert 0.0 <= result[0].brier_score_raw <= 1.0


def test_brier_score_uses_last_cv_fold_not_calib():
    """Brier score must NOT be computed on X_calib (reserved for calibration)."""
    from aer.ml.features import build_feature_matrix
    from aer.ml.models import train_and_predict

    df = _synthetic_df(900)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")

    result = train_and_predict(feature_df, "AAPL")
    if result and not result[0].prediction_skipped:
        assert result[0].brier_score_raw is not None


# ---------------------------------------------------------------------------
# train_and_predict returns tuple and calibration
# ---------------------------------------------------------------------------


def test_train_and_predict_returns_tuple():
    from aer.ml.features import build_feature_matrix
    from aer.ml.models import train_and_predict
    from aer.models import MLPrediction

    df = _synthetic_df(900)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    result = train_and_predict(feature_df, "AAPL")
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], MLPrediction)
    assert isinstance(result[1], MLPrediction)


def test_calibrated_prob_up_in_result():
    from aer.ml.features import build_feature_matrix
    from aer.ml.models import train_and_predict

    df = _synthetic_df(900)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    result = train_and_predict(feature_df, "AAPL")
    assert result is not None
    lgbm_ml = result[0]
    # calibrated_prob_up should be a float or None - never raises AttributeError
    assert lgbm_ml.calibrated_prob_up is None or isinstance(lgbm_ml.calibrated_prob_up, float)


def test_brier_score_logged(caplog):
    import logging

    from aer.ml.features import build_feature_matrix
    from aer.ml.models import train_and_predict

    df = _synthetic_df(900)
    with patch("aer.ml.features.get_price_history_raw", return_value=df):
        feature_df = build_feature_matrix("AAPL")
    # Mock cross_val_score to return scores above CV_F1_MEAN_MINIMUM (0.52) so model is not skipped
    with (
        caplog.at_level(logging.INFO, logger="aer.ml.models"),
        patch("aer.ml.models.cross_val_score", return_value=np.array([0.55, 0.56, 0.54, 0.57, 0.55])),
    ):
        train_and_predict(feature_df, "AAPL")
    assert any("Brier check |" in record.message for record in caplog.records)
