# ADR 0004: ML gate with purged cross-validation and abstention

## Status

Accepted

## Context

Labels look five bars ahead, so neighbouring rows share information. A random split, or calibration on rows a model was fitted on, would flatter the model. Early releases contained three such defects, which the README lists and which later releases fixed with a test each.

## Decision

`src/aer/ml/models.py` fits LightGBM and logistic regression. Labelled rows are ordered in time as a cross-validation block, 10 dropped purge rows, and a holdout block of the last 15%. Five time-series folds use a purge gap of 10 bars and a test size of 55 bars. LightGBM is selected only when its mean cross-validation F1 is strictly higher; otherwise logistic regression is. A Platt calibrator is fitted on the pooled out-of-fold LightGBM probabilities. The holdout block scores a Brier diagnostic and never enters a fit. The gate abstains when the selected model's mean F1 is below 0.52 or its standard deviation exceeds 0.15, and fewer than 400 training bars return no fit.

## Alternatives considered

- **Random k-fold splits.** Overlapping labels would cross fold boundaries.
- **Calibrating on training rows.** The probabilities would be optimistic, which was one of the early defects.
- **Always predicting.** A model with no skill would still issue calls.

## Consequences

- Positive. Each leakage check listed under Enforced by has a test, and an abstention is a recorded outcome.
- Negative. In the control the gate passed in 4 of 46 seeds at zero signal and 11 of 46 at the strongest signal, a descriptive count. Only the LightGBM member is Platt-calibrated. A conformal gate is the planned replacement.

## Enforced by

- `tests/ml/test_leakage.py` (`test_label_is_forward_return_at_horizon`, `test_features_do_not_depend_on_future_bars`, `test_purge_gap_geometry`)
- `tests/test_models_fixes.py` (`test_split_leaves_purge_gap_and_holdout_is_the_last_fifteen_percent`, `test_no_holdout_or_gap_row_is_used_for_fitting`, `test_calibrator_is_fitted_on_out_of_fold_probabilities`)
- `tests/test_ml.py` (`test_high_std_returns_skipped`)
