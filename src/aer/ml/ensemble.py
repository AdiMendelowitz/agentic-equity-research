"""F1-weighted soft-vote ensemble, and the sigmoid calibration class the training code uses.

- _SC is sklearn's private _SigmoidCalibration. aer.ml.models fits it on pooled out-of-fold LightGBM
  probabilities (_fit_platt_calibrator); no calibrator is fitted in this module.
- compute_ensemble: F1-weighted soft vote combining calibrated LightGBM prob and logistic prob.
"""

from typing import Any

try:
    from sklearn.calibration import _SigmoidCalibration as _SC  # private API - sklearn 1.8 verified
except ImportError:
    _SC = None  # guard against future sklearn API removal

from aer.models import MLPrediction


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
