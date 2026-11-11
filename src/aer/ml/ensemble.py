"""Platt calibration and F1-weighted soft-vote ensemble.

Design:
- _fit_sigmoid_calibrator: fits sklearn._SigmoidCalibration on LightGBM X_calib predictions.
  Uses the private _SigmoidCalibration directly. CalibratedClassifierCV(cv='prefit') is deprecated
  since sklearn 1.6 in favour of FrozenEstimator; moving to that public API is planned.
- compute_ensemble: F1-weighted soft vote combining calibrated LightGBM prob and logistic prob.
"""

import logging
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

try:
    from sklearn.calibration import _SigmoidCalibration as _SC  # private API - sklearn 1.8 verified
except ImportError:
    _SC = None  # guard against future sklearn API removal

from aer.models import MLPrediction

logger = logging.getLogger(__name__)


def _fit_sigmoid_calibrator(
    pipe: Pipeline,
    X_calib: Any,
    y_calib: np.ndarray,
) -> "Any | None":
    """Fit a Platt sigmoid calibrator on the calibration set. Returns None on failure."""
    if _SC is None:
        logger.warning("_SigmoidCalibration unavailable — sklearn API changed; calibration skipped")
        return None
    # Degenerate guard: must have both classes present BEFORE calling sc.fit()
    if len(np.unique(y_calib)) < 2:
        logger.warning("Calibration set has only one class — degenerate; calibration skipped")
        return None
    try:
        raw_probs = pipe.predict_proba(X_calib)[:, 1]
        sc = _SC()
        sc.fit(raw_probs, y_calib)
        return sc
    except Exception as exc:
        logger.warning("Calibration fit failed (%s): %s", type(exc).__name__, exc)
        return None


def compute_ensemble(
    lgbm_pred: MLPrediction,
    logistic_pred: MLPrediction,
    f1_lgbm: float,
    f1_logistic: float,
) -> dict[str, Any]:
    """Return dict of ensemble fields for MLPrediction.model_copy(update={...}).

    Returns {} if either model is skipped or required probabilities are None.
    The caller should do: ml = ml.model_copy(update=compute_ensemble(...)) if ensemble != {}.
    """
    if lgbm_pred.prediction_skipped or logistic_pred.prediction_skipped:
        return {}

    calibrated = lgbm_pred.calibrated_prob_up
    lgbm_prob = calibrated if calibrated is not None else lgbm_pred.prob_up_raw
    logistic_prob = logistic_pred.prob_up_raw
    if lgbm_prob is None or logistic_prob is None:
        return {}

    denominator = f1_lgbm + f1_logistic
    weight_lgbm = f1_lgbm / denominator if denominator > 0 else 0.5
    weight_logistic = 1.0 - weight_lgbm

    ensemble_prob = weight_lgbm * lgbm_prob + weight_logistic * logistic_prob
    return {
        "ensemble_prob_up": round(float(ensemble_prob), 6),
        "ensemble_direction": "UP" if ensemble_prob >= 0.5 else "DOWN",
    }
