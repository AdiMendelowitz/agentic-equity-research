"""ML model training, selection, and prediction.

Design notes:
- class_weight="balanced" on logistic; is_unbalance=True on LightGBM
- Labelled rows in time order: CV block | PURGE_GAP_BARS dropped rows | holdout block. Labels look
  PREDICTION_HORIZON_DAYS bars ahead, so the dropped rows stop CV labels overlapping the holdout.
- One manual TimeSeriesSplit loop over the CV block records per-fold macro F1 and the out-of-fold probabilities
  of both models. The reported Brier score is out of fold, pooled over every CV test row of the selected model.
- The Platt calibrator is fitted on the pooled out-of-fold LightGBM probabilities (cross-fitted). The holdout
  block is never used to fit anything; it only scores the Brier check (LightGBM, logistic, ensemble).
- fit_ensemble returns a reusable FittedEnsemble (both final models refitted on the whole CV block, whether or
  not the gate passes); train_and_predict is built on top of it.
- Combined skip: mean_f1 < CV_F1_MEAN_MINIMUM OR cv_f1_std > CV_F1_STD_HARD_LIMIT
- model_used="none" when prediction_skipped=True
"""

import logging
import os
import sys
import threading
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.model_selection import TimeSeriesSplit
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
from aer.ml.ensemble import _SC
from aer.models import MLPrediction, _utcnow

logger = logging.getLogger(__name__)

_MODEL_WRITE_LOCK = threading.Lock()

HOLDOUT_FRACTION: float = 0.15


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


@dataclass(frozen=True)
class FittedEnsemble:
    """Both final pipelines, the cross-fitted Platt calibrator and the diagnostics of one fit.

    Both pipelines are refitted on the whole CV block. The holdout block never touches a fit.
    """

    features: tuple[str, ...]
    lgbm: Pipeline
    logistic: Pipeline
    calibrator: Any | None
    cv_f1_lgbm: float
    cv_f1_logistic: float
    cv_f1_std_lgbm: float
    cv_f1_std_logistic: float
    model_used: str
    gate_passed: bool
    oof_brier: float | None
    holdout_brier: dict[str, float] | None
    trained_on_bars: int
    up_rate: float

    @property
    def weight_lgbm(self) -> float:
        total = self.cv_f1_lgbm + self.cv_f1_logistic
        return self.cv_f1_lgbm / total if total > 0 else 0.5

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        """F1-weighted soft vote of the calibrated LightGBM and the logistic probabilities of the up class."""
        frame = X[list(self.features)]
        lgbm_prob = self.lgbm.predict_proba(frame)[:, 1]
        if self.calibrator is not None:
            lgbm_prob = np.asarray(self.calibrator.predict(lgbm_prob), dtype=float)
        logistic_prob = self.logistic.predict_proba(frame)[:, 1]
        weight = self.weight_lgbm
        return weight * lgbm_prob + (1.0 - weight) * logistic_prob


@dataclass(frozen=True)
class _CvResult:
    f1_scores: np.ndarray
    oof_index: np.ndarray
    oof_probs: np.ndarray


def _split_cv_holdout(n_rows: int) -> tuple[slice, slice]:
    """Positional slices of the CV block and the holdout block, with PURGE_GAP_BARS rows dropped between them."""
    n_holdout = int(n_rows * HOLDOUT_FRACTION)
    holdout_start = n_rows - n_holdout
    cv_stop = max(holdout_start - PURGE_GAP_BARS, 0)
    return slice(0, cv_stop), slice(holdout_start, n_rows)


def _make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)]).set_output(transform="pandas")


def _cross_validate(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> _CvResult:
    """Fit a clone of pipe per fold; return macro F1 per fold and the pooled out-of-fold up probabilities."""
    f1_scores: list[float] = []
    oof_index: list[np.ndarray] = []
    oof_probs: list[np.ndarray] = []
    for train_idx, test_idx in splits:
        fold_pipe = clone(pipe).set_output(transform="pandas")
        fold_pipe.fit(X.iloc[train_idx], y[train_idx])
        X_test = X.iloc[test_idx]
        f1_scores.append(float(f1_score(y[test_idx], fold_pipe.predict(X_test), average="macro")))
        oof_index.append(test_idx)
        oof_probs.append(fold_pipe.predict_proba(X_test)[:, 1])
    return _CvResult(np.asarray(f1_scores), np.concatenate(oof_index), np.concatenate(oof_probs))


def _fit_platt_calibrator(probs: np.ndarray, y: np.ndarray) -> Any | None:
    """Fit a Platt sigmoid on probabilities and true labels. Returns None when degenerate or on failure."""
    if _SC is None:
        logger.warning("_SigmoidCalibration unavailable (sklearn API changed); calibration skipped")
        return None
    if len(np.unique(y)) < 2:
        logger.warning("Calibration set has only one class (degenerate); calibration skipped")
        return None
    try:
        sc = _SC()
        sc.fit(probs, y)
        return sc
    except Exception as exc:
        logger.warning("Calibration fit failed (%s): %s", type(exc).__name__, exc)
        return None


def _holdout_brier(ens: FittedEnsemble, X_holdout: pd.DataFrame, y_holdout: np.ndarray) -> dict[str, float] | None:
    if len(y_holdout) == 0:
        return None
    try:
        frame = X_holdout[list(ens.features)]
        return {
            "lgbm": float(brier_score_loss(y_holdout, ens.lgbm.predict_proba(frame)[:, 1])),
            "logistic": float(brier_score_loss(y_holdout, ens.logistic.predict_proba(frame)[:, 1])),
            "ensemble": float(brier_score_loss(y_holdout, ens.predict_proba_up(X_holdout))),
        }
    except Exception as exc:
        logger.warning("Holdout Brier check failed: %s", exc)
        return None


def fit_ensemble(feature_df: pd.DataFrame) -> FittedEnsemble | None:
    """Cross-validate, calibrate on out-of-fold probabilities and refit both models on the CV block.

    Returns None when fewer than MIN_TRAINING_BARS rows carry a label. Both final models are always fitted,
    including when the gate fails, so that callers can score the model regardless of the gate. Writes no files.
    """
    features = tuple(c for c in FEATURE_COLS if c in feature_df.columns)
    train_df = feature_df.dropna(subset=["label"])
    if len(train_df) < MIN_TRAINING_BARS:
        return None

    X_all = train_df[list(features)]
    y_all: np.ndarray = train_df["label"].to_numpy(dtype=int)
    cv_slice, holdout_slice = _split_cv_holdout(len(X_all))
    X_cv, y_cv = X_all.iloc[cv_slice], y_all[cv_slice]
    X_holdout, y_holdout = X_all.iloc[holdout_slice], y_all[holdout_slice]

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS, gap=PURGE_GAP_BARS, test_size=CV_TEST_SIZE_BARS)
    splits = list(tscv.split(X_cv))

    logreg_pipe = _make_pipeline(_make_logistic())
    lgbm_pipe = _make_pipeline(_make_lgbm())
    logreg_cv = _cross_validate(logreg_pipe, X_cv, y_cv, splits)
    lgbm_cv = _cross_validate(lgbm_pipe, X_cv, y_cv, splits)

    # Model selection: higher mean F1 wins; tie → logistic (lighter, faster)
    use_lgbm = lgbm_cv.f1_scores.mean() > logreg_cv.f1_scores.mean()
    model_used = "lightgbm" if use_lgbm else "logistic"
    best_cv = lgbm_cv if use_lgbm else logreg_cv
    best_mean = float(best_cv.f1_scores.mean())
    best_std = float(best_cv.f1_scores.std())
    gate_passed = not (best_mean < CV_F1_MEAN_MINIMUM or best_std > CV_F1_STD_HARD_LIMIT)

    oof_brier: float | None = None
    if len(best_cv.oof_index) > 0:
        oof_brier = float(brier_score_loss(y_cv[best_cv.oof_index], best_cv.oof_probs))

    calibrator = _fit_platt_calibrator(lgbm_cv.oof_probs, y_cv[lgbm_cv.oof_index])

    logreg_pipe.fit(X_cv, y_cv)
    lgbm_pipe.fit(X_cv, y_cv)

    ens = FittedEnsemble(
        features=features,
        lgbm=lgbm_pipe,
        logistic=logreg_pipe,
        calibrator=calibrator,
        cv_f1_lgbm=float(lgbm_cv.f1_scores.mean()),
        cv_f1_logistic=float(logreg_cv.f1_scores.mean()),
        cv_f1_std_lgbm=float(lgbm_cv.f1_scores.std()),
        cv_f1_std_logistic=float(logreg_cv.f1_scores.std()),
        model_used=model_used,
        gate_passed=gate_passed,
        oof_brier=oof_brier,
        holdout_brier=None,
        trained_on_bars=len(X_cv),
        up_rate=float(y_cv.mean()),
    )
    return replace(ens, holdout_brier=_holdout_brier(ens, X_holdout, y_holdout))


def train_and_predict(feature_df: pd.DataFrame, ticker: str) -> tuple["MLPrediction", "MLPrediction"] | None:
    """Train logistic + LightGBM on historical labelled data; return prediction for latest bar.

    Returns (lgbm_ml, logistic_ml) tuple - both MLPrediction objects for ensemble use.
    feature_df: output of build_feature_matrix() - may contain NaN in 'label' column
                for the most recent bars (no forward return available yet).
    The reported Brier score is out of fold; the Brier check is scored on a holdout block that no fit has seen.
    """
    available_features = [c for c in FEATURE_COLS if c in feature_df.columns]
    n_labelled = len(feature_df.dropna(subset=["label"]))

    ens = fit_ensemble(feature_df)
    if ens is None:
        logger.warning("%s: only %d labeled rows — skipping ML prediction", ticker, n_labelled)
        _skipped = MLPrediction(
            ticker=ticker,
            prediction_skipped=True,
            model_used="none",
            trained_on_bars=n_labelled,
            n_features=len(available_features),
        )
        return _skipped, _skipped

    model_used = ens.model_used
    mean_f1 = ens.cv_f1_lgbm if model_used == "lightgbm" else ens.cv_f1_logistic
    std_f1 = ens.cv_f1_std_lgbm if model_used == "lightgbm" else ens.cv_f1_std_logistic
    up_rate = ens.up_rate
    n_bars = ens.trained_on_bars

    # Combined skip criterion: mean too low OR std too high
    if not ens.gate_passed:
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
            trained_on_bars=n_bars,
            n_features=len(available_features),
        )
        return _skipped, _skipped

    lgbm_pipe = ens.lgbm
    logreg_pipe = ens.logistic
    best_pipe = lgbm_pipe if model_used == "lightgbm" else logreg_pipe
    calibrator = ens.calibrator

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

    brier_score_raw = ens.oof_brier

    # Verification on the holdout block: no fit, including the calibrator, has seen these rows
    if ens.holdout_brier is not None:
        brier_lgbm = ens.holdout_brier["lgbm"]
        brier_logistic = ens.holdout_brier["logistic"]
        brier_ensemble = ens.holdout_brier["ensemble"]
        brier_check_passed = brier_ensemble < min(brier_lgbm, brier_logistic)
        logger.info(
            "Brier check | %s | brier_lgbm=%.4f brier_logistic=%.4f brier_ensemble=%.4f passed=%s",
            ticker,
            brier_lgbm,
            brier_logistic,
            brier_ensemble,
            brier_check_passed,
        )
    else:
        logger.warning("Brier check unavailable for %s", ticker)

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
        n_bars,
    )

    # Build per-model MLPrediction objects for ensemble
    lgbm_ml = MLPrediction(
        ticker=ticker,
        direction="UP" if lgbm_prob_up_raw >= 0.5 else "DOWN",
        prob_up_raw=lgbm_prob_up_raw,
        calibrated_prob_up=calibrated_prob_up,
        cv_f1_macro=ens.cv_f1_lgbm,
        cv_f1_std=ens.cv_f1_std_lgbm,
        model_used="lightgbm",
        trained_on_bars=n_bars,
        n_features=len(available_features),
        class_balance_up=up_rate,
        brier_score_raw=brier_score_raw if model_used == "lightgbm" else None,
        as_of=_utcnow(),
    )
    logistic_ml = MLPrediction(
        ticker=ticker,
        direction="UP" if logreg_prob_up_raw >= 0.5 else "DOWN",
        prob_up_raw=logreg_prob_up_raw,
        cv_f1_macro=ens.cv_f1_logistic,
        cv_f1_std=ens.cv_f1_std_logistic,
        model_used="logistic",
        trained_on_bars=n_bars,
        n_features=len(available_features),
        class_balance_up=up_rate,
        brier_score_raw=brier_score_raw if model_used == "logistic" else None,
        as_of=_utcnow(),
    )
    return lgbm_ml, logistic_ml
