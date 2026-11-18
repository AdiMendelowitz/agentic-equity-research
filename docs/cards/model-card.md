# Model card

This card follows the spirit of Model Cards for Model Reporting (Mitchell et al.). It covers the ML layer in `src/aer/ml/`.

## Model details

- **Models.** LightGBM and logistic regression (`models.py`), each behind a standard scaler. LightGBM uses 200 estimators, 31 leaves, minimum child samples 30, learning rate 0.05, column sample 0.8, row subsample 1.0 with bagging off and `is_unbalance=True`. Logistic regression uses `class_weight="balanced"`. The seed is 42 (`constants.py`).
- **Features.** The columns in `FEATURE_COLS` (`constants.py`), built by `features.py`: 37 indicator and rolling-window columns such as RSI, MACD, Bollinger bands, moving averages, ATR, ADX and 52-week high and low distances. Two Ichimoku columns are excluded, one because it puts later closes on earlier rows and one because its row alignment was not verified.
- **Label.** One when the five-day forward return exceeds the round-trip cost of 10 basis points, otherwise zero, on unadjusted closes with a guard for stock splits (`features.py`, `PREDICTION_HORIZON_DAYS = 5`, `ROUND_TRIP_COST_BPS = 10.0`).
- **Validation.** Labelled rows are ordered in time as a cross-validation block, 10 dropped rows (`PURGE_GAP_BARS`) and a holdout block of 15% (`HOLDOUT_FRACTION`). Cross-validation uses `TimeSeriesSplit` with `N_CV_SPLITS = 5`, a gap of `PURGE_GAP_BARS` and `CV_TEST_SIZE_BARS = 55`. The holdout block is never used to fit anything and only scores a Brier diagnostic.
- **Calibration.** A Platt sigmoid is fitted on the pooled out-of-fold LightGBM probabilities. Only the LightGBM member is calibrated. The ensemble probability is an F1-weighted soft vote of the LightGBM probability, calibrated when a calibrator was fitted, and the logistic probability.
- **Gate.** The model with the higher mean cross-validation macro F1 is selected, with ties going to logistic regression. The gate passes unless that model's mean F1 is below `CV_F1_MEAN_MINIMUM` (0.52) or its F1 standard deviation exceeds `CV_F1_STD_HARD_LIMIT` (0.15). When it fails, the layer abstains. `selected_prediction` returns the member the gate validated by the same rule, and the brief carries that prediction with the ensemble probability beside it.

## Intended use

A research demonstration of an evaluation method. It is not investment advice and not a trading system. By default it runs offline on synthetic data and makes no model call.

## Evaluation

**Pre-registered control, version 3.** The control scores the two-model ensemble on 46 fresh seeds per signal strength on synthetic prices, whether or not the gate passed.

| Signal strength s | Ensemble balanced accuracy, every seed | Pre-registered test | Verdict | Gate passed |
| --- | --- | --- | --- | --- |
| 0.0 | 0.516 | 90% interval 0.510 to 0.522 inside 0.47 to 0.53 | equivalent to 0.50 within 0.03: met | 4 of 46 seeds |
| 0.3 | 0.530 | lower bound 0.519, Holm-adjusted p below 0.001 | above 0.50: met | 5 of 46 seeds |
| 0.6 | 0.565 | lower bound 0.547, Holm-adjusted p below 0.001 | above 0.50: met | 11 of 46 seeds |

The superiority tests compare each strength with 0.50, and the 0.03 margin applies only to the equivalence test at zero signal. The lower bound at 0.3 lies inside the zero-signal interval, and no test between signal strengths was pre-registered, so the table supports comparisons with 0.50 only.

**Recorded output.** 314 of 629 resolved calls were correct (0.499, 95% Wilson interval 0.460 to 0.538), counting one decision per ticker and day. Detecting 0.55 against 0.50 would need about 780 independent calls under a two-sided test at the 5% level with 80% power, by the normal approximation in `src/aer/evaluation/power.py`. A rule that always says down scores 350 of 629 (0.556) on the same calls. The ML gate predicted on 139 of 2,112 rows, and 64 of the 120 resolved predictions were correct (0.533). These recorded predictions are the LightGBM member's direction, while the control scores the calibrated ensemble. The figures were corrected on 17 November 2026.

## Effective configuration

The LightGBM classifier sets `subsample=1.0` and leaves `subsample_freq` at 0, so every tree is grown on all training rows. Earlier releases set `subsample=0.8`, which LightGBM ignores while `subsample_freq` is 0, its documented default. No fit in this repository, the pre-registered control included, used row subsampling, and the change to 1.0 moves no fitted model and no published figure. `tests/ml/test_lgbm_config.py` pins both settings and checks that a fit with `subsample=0.5` and no frequency grows the same trees as the stated configuration.

## Limitations

- Samples are small, with 629 resolved calls and 120 resolved ML predictions.
- The control uses one synthetic signal family. It shows the harness can find that signal and does not show that real prices carry one.
- In version 3 the model scores slightly above 0.50 on pure noise, within the pre-registered margin, and the cause is not identified.
- Only the LightGBM member is Platt-calibrated.
- Recorded calls are scored by direction at zero on adjusted closes, while the ML label is a five-day return above a 10 basis point cost on unadjusted closes.
- The 139 published ML predictions were recorded under an earlier rule that kept the LightGBM member's direction.
- Briefs are scored against a five-day move, while the underlying thesis spans weeks to months.
- Live mode is absent.

## Ethical and risk considerations

The outputs give no advice and should not inform any financial decision. The repository has no live mode. The demo data are synthetic and no real company appears in it. Null and corrected results are published with the figures they replace.
