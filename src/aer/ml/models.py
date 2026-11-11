"""ML model training, selection, and prediction.

Design notes:
- class_weight="balanced" on logistic; is_unbalance=True on LightGBM
- Brier score on last CV fold test set - X_calib is untouched (reserved for calibration)
- Combined skip: mean_f1 < CV_F1_MEAN_MINIMUM OR cv_f1_std > CV_F1_STD_HARD_LIMIT
- model_used="none" when prediction_skipped=True
"""

import logging
import os
import sys
import threading
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aer.ml.constants import (
    CV_F1_MEAN_MINIMUM,
    CV_F1_STD_HARD_LIMIT,
    CV_TEST_SIZE_BARS,
    FEATURE_COLS,
    LGBM_COLSAMPLE_BYTREE,
    LGBM_LEARNING_RATE,
    LGBM_MIN_CHILD_SAMPLES,
    LGBM_N_ESTIMATORS,
    LGBM_NUM_LEAVES,
    LGBM_SUBSAMPLE,
    MIN_TRAINING_BARS,
    MODEL_DIR,
    MODEL_RETENTION_DAYS,
    N_CV_SPLITS,
    PURGE_GAP_BARS,
    SEED,
)
from aer.models import MLPrediction, _utcnow

logger = logging.getLogger(__name__)

_MODEL_WRITE_LOCK = threading.Lock()


def _make_lgbm() -> Any:
    import lightgbm as lgb  # local import - heavy dep

    return lgb.LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        num_leaves=LGBM_NUM_LEAVES,
        min_child_samples=LGBM_MIN_CHILD_SAMPLES,
        learning_rate=LGBM_LEARNING_RATE,
        subsample=LGBM_SUBSAMPLE,
        colsample_bytree=LGBM_COLSAMPLE_BYTREE,
        random_state=SEED,
        is_unbalance=True,  # handle class imbalance in LightGBM
        verbose=-1,
    )


def _make_logistic() -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        random_state=SEED,
        class_weight="balanced",  # handle class imbalance in logistic regression
    )


def _try_lock_file(fileobj) -> bool:
    """Non-blocking exclusive lock on an open file. Cross-platform (msvcrt on Windows,
    fcntl on POSIX). Returns True if acquired, False if held by another process.

    Previously msvcrt-only, which raised ModuleNotFoundError on Linux (caught by the bare
    except → save silently returned False on every non-Windows host, incl. CI).
    """
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fileobj.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fileobj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock_file(fileobj) -> None:
    """Release a lock acquired by _try_lock_file. Best-effort."""
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fileobj.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fileobj.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _atomic_save_model(ticker: str, model_path: Path, payload: object) -> bool:
    """Save model atomically using threading + per-ticker file lock. Returns True on success."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".tmp")
    lock_path = model_path.with_suffix(".lock")

    with _MODEL_WRITE_LOCK:
        try:
            with open(lock_path, "w") as lf:
                if not _try_lock_file(lf):
                    logger.warning("Model write lock held by another process for %s — skipping save", ticker)
                    return False
                try:
                    joblib.dump(payload, tmp_path)
                    os.replace(tmp_path, model_path)
                finally:
                    _unlock_file(lf)
        except Exception as exc:
            logger.warning("Model save failed for %s: %s", ticker, exc)
            return False
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    return True


def _cleanup_old_models(ticker: str) -> None:
    if not MODEL_DIR.exists():
        return
    cutoff = date.today().toordinal() - MODEL_RETENTION_DAYS
    for path in MODEL_DIR.glob(f"{ticker}_*.joblib"):
        try:
            date_part = path.stem.split("_")[-1]
            file_date = date.fromisoformat(date_part).toordinal()
            if file_date < cutoff:
                path.unlink()
        except (ValueError, OSError):
            pass


def train_and_predict(feature_df: pd.DataFrame, ticker: str) -> tuple["MLPrediction", "MLPrediction"] | None:
    """Train logistic + LightGBM on historical labeled data; return prediction for latest bar.

    Returns (lgbm_ml, logistic_ml) tuple - both MLPrediction objects for ensemble use.
    feature_df: output of build_feature_matrix() - may contain NaN in 'label' column
                for the most recent bars (no forward return available yet).
    """
    available_features = [c for c in FEATURE_COLS if c in feature_df.columns]
    train_df = feature_df.dropna(subset=["label"])

    if len(train_df) < MIN_TRAINING_BARS:
        logger.warning("%s: only %d labeled rows — skipping ML prediction", ticker, len(train_df))
        _skipped = MLPrediction(
            ticker=ticker,
            prediction_skipped=True,
            model_used="none",
            trained_on_bars=len(train_df),
            n_features=len(available_features),
        )
        return _skipped, _skipped

    X_all = train_df[available_features]
    y_all = train_df["label"].values.astype(int)

    # Reserve 15% calibration set for Platt calibration
    n_calib = int(len(X_all) * 0.15)
    X_cv, y_cv = X_all.iloc[:-n_calib], y_all[:-n_calib]
    X_calib = X_all.iloc[-n_calib:]
    y_calib = y_all[-n_calib:]

    up_rate = float(y_cv.mean())
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS, gap=PURGE_GAP_BARS, test_size=CV_TEST_SIZE_BARS)

    logreg_pipe = Pipeline([("scaler", StandardScaler()), ("clf", _make_logistic())]).set_output(transform="pandas")
    lgbm_pipe = Pipeline([("scaler", StandardScaler()), ("clf", _make_lgbm())]).set_output(transform="pandas")

    logreg_scores = cross_val_score(logreg_pipe, X_cv, y_cv, cv=tscv, scoring="f1_macro")
    lgbm_scores = cross_val_score(lgbm_pipe, X_cv, y_cv, cv=tscv, scoring="f1_macro")

    # Model selection: higher mean F1 wins; tie → logistic (lighter, faster)
    if lgbm_scores.mean() > logreg_scores.mean():
        best_pipe = lgbm_pipe
        model_used = "lightgbm"
        best_scores = lgbm_scores
    else:
        best_pipe = logreg_pipe
        model_used = "logistic"
        best_scores = logreg_scores

    mean_f1 = float(best_scores.mean())
    std_f1 = float(best_scores.std())

    # Combined skip criterion: mean too low OR std too high
    if mean_f1 < CV_F1_MEAN_MINIMUM or std_f1 > CV_F1_STD_HARD_LIMIT:
        logger.info(
            "%s: prediction skipped — mean_f1=%.3f (min %.2f), std_f1=%.3f (max %.2f)",
            ticker,
            mean_f1,
            CV_F1_MEAN_MINIMUM,
            std_f1,
            CV_F1_STD_HARD_LIMIT,
        )
        _skipped = MLPrediction(
            ticker=ticker,
            prediction_skipped=True,
            model_used="none",  # explicit "none", not the "lightgbm" default
            cv_f1_macro=mean_f1,
            cv_f1_std=std_f1,
            class_balance_up=up_rate,
            trained_on_bars=len(X_cv),
            n_features=len(available_features),
        )
        return _skipped, _skipped

    # Fit BOTH pipelines - ensemble requires both to predict on X_pred
    logreg_pipe.fit(X_cv, y_cv)
    lgbm_pipe.fit(X_cv, y_cv)

    # Platt calibration on LightGBM
    from aer.ml.ensemble import _fit_sigmoid_calibrator  # local import - after skip check

    calibrator = _fit_sigmoid_calibrator(lgbm_pipe, X_calib, y_calib)

    # Predict on the latest available bar (last row of feature_df regardless of label)
    X_pred = feature_df[available_features].iloc[[-1]]

    lgbm_probs_pred = lgbm_pipe.predict_proba(X_pred)
    lgbm_prob_up_raw = float(lgbm_probs_pred[0][1])

    logreg_probs_pred = logreg_pipe.predict_proba(X_pred)
    logreg_prob_up_raw = float(logreg_probs_pred[0][1])

    # Apply calibrator to LightGBM prediction
    calibrated_prob_up: float | None = None
    if calibrator is not None:
        try:
            calibrated_prob_up = float(calibrator.predict(np.array([lgbm_prob_up_raw]))[0])
        except Exception as _calib_exc:
            logger.warning("Calibrator predict failed for %s: %s", ticker, _calib_exc)

    # Brier score on last CV fold test set - NOT on X_calib
    brier_score_raw: float | None = None
    last_fold = list(tscv.split(X_cv))[-1]
    X_last_test = X_cv.iloc[last_fold[1]]
    y_last_test = y_cv[last_fold[1]]
    if len(X_last_test) > 0:
        last_probs = best_pipe.predict_proba(X_last_test)[:, 1]
        brier_score_raw = float(brier_score_loss(y_last_test, last_probs))

    # Verification: ensemble Brier score vs individual models on X_calib
    try:
        lgbm_probs_calib = lgbm_pipe.predict_proba(X_calib)[:, 1]
        logreg_probs_calib = logreg_pipe.predict_proba(X_calib)[:, 1]
        calibrated_probs_calib = calibrator.predict(lgbm_probs_calib) if calibrator is not None else lgbm_probs_calib
        _f1_sum = float(lgbm_scores.mean()) + float(logreg_scores.mean())
        _w_lgbm = float(lgbm_scores.mean()) / _f1_sum if _f1_sum > 0 else 0.5
        _w_logistic = 1.0 - _w_lgbm
        ensemble_probs_calib = _w_lgbm * calibrated_probs_calib + _w_logistic * logreg_probs_calib
        brier_lgbm = float(brier_score_loss(y_calib, lgbm_probs_calib))
        brier_logistic = float(brier_score_loss(y_calib, logreg_probs_calib))
        brier_ensemble = float(brier_score_loss(y_calib, ensemble_probs_calib))
        brier_check_passed = brier_ensemble < min(brier_lgbm, brier_logistic)
        logger.info(
            "Brier check | %s | brier_lgbm=%.4f brier_logistic=%.4f brier_ensemble=%.4f passed=%s",
            ticker,
            brier_lgbm,
            brier_logistic,
            brier_ensemble,
            brier_check_passed,
        )
    except Exception as _brier_exc:
        logger.warning("Brier check failed for %s: %s", ticker, _brier_exc)

    # Persist model with calibrator key (backward-compatible; None if calibration failed)
    today_str = date.today().isoformat()
    model_path = MODEL_DIR / f"{ticker}_{today_str}.joblib"
    _atomic_save_model(
        ticker,
        model_path,
        {"pipeline": best_pipe, "features": available_features, "calibrator": calibrator},
    )
    _cleanup_old_models(ticker)

    logger.info(
        "%s: model=%s, mean_f1=%.3f, std_f1=%.3f, direction=%s, prob_up=%.3f, bars=%d",
        ticker,
        model_used,
        mean_f1,
        std_f1,
        "UP" if lgbm_prob_up_raw >= 0.5 else "DOWN",
        lgbm_prob_up_raw,
        len(X_cv),
    )

    # Build per-model MLPrediction objects for ensemble
    lgbm_ml = MLPrediction(
        ticker=ticker,
        direction="UP" if lgbm_prob_up_raw >= 0.5 else "DOWN",
        prob_up_raw=lgbm_prob_up_raw,
        calibrated_prob_up=calibrated_prob_up,
        cv_f1_macro=float(lgbm_scores.mean()),
        cv_f1_std=float(lgbm_scores.std()),
        model_used="lightgbm",
        trained_on_bars=len(X_cv),
        n_features=len(available_features),
        class_balance_up=up_rate,
        brier_score_raw=brier_score_raw if model_used == "lightgbm" else None,
        as_of=_utcnow(),
    )
    logistic_ml = MLPrediction(
        ticker=ticker,
        direction="UP" if logreg_prob_up_raw >= 0.5 else "DOWN",
        prob_up_raw=logreg_prob_up_raw,
        cv_f1_macro=float(logreg_scores.mean()),
        cv_f1_std=float(logreg_scores.std()),
        model_used="logistic",
        trained_on_bars=len(X_cv),
        n_features=len(available_features),
        class_balance_up=up_rate,
        brier_score_raw=brier_score_raw if model_used == "logistic" else None,
        as_of=_utcnow(),
    )
    return lgbm_ml, logistic_ml
