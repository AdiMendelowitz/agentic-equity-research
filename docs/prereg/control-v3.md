# Pre-registration: detection control, version 3

Specification identifier: control-v3. The constants below are the single source of truth and are mirrored in `src/aer/demo/prereg_v3.py`. A test checks that the two agree.

## Purpose

Version 2 asked whether a classifier trained on synthetic prices with a known injected signal beats chance. This rerun asks the same question of the deployed model, with a formal statistical specification fixed before the run. Version 2 stays unchanged and reproducible.

## What changed from version 2, and why

- Version 2 scored one uncalibrated classifier per seed, not the deployed model. Version 3 scores the deployed calibrated ensemble, with probabilities thresholded at 0.50.
- A purge gap now separates the training block from the holdout block, so five-day labels do not overlap at the boundary.
- Calibration uses out-of-fold predictions.
- Seeds start at 1000, so no seed is shared with version 2 (seeds 0 to 19).
- Informal expectations are replaced by the tests below.

## Data-generating process and fixed design

- Synthetic prices with constant volatility and an injected signal; generator in `src/aer/demo/generate.py`, unchanged.
- Signals: 0.0, 0.3, 0.6.
- 1,500 sessions per series; the last 20% of rows form the out-of-sample block after a purge gap.
- The oracle uses the true conditional mean on the same out-of-sample rows and labels as the model.

## Specification constants

- DELTA = 0.03
- ALPHA = 0.05
- POWER = 0.80
- SD_UPPER_CONFIDENCE = 0.80
- MIN_SEEDS = 20
- SEED_OFFSET = 1000
- SIGNALS = 0.0, 0.3, 0.6

DELTA is the smallest balanced-accuracy edge worth claiming. It is about one fifth of the gap between chance (0.50) and the analytic ceiling at signal 0.3 (0.641, from `analytic_ceiling` in `src/aer/demo/control.py`). It was chosen from that closed form and not from the pilot's own effect. The version 2 pilot supplies only the standard deviation. It measured a different model (one uncalibrated classifier), which is one reason each standard deviation is inflated to its upper confidence limit before use.

The hash published before the run covers this file, `src/aer/demo/prereg_v3.py` and `src/aer/demo/control_v3.py`.

## Sample size rule

Power is 0.80. Each pilot standard deviation is inflated to its upper 80% one-sided confidence limit. The seed count is the larger of the equivalence count (zero signal, at alpha) and the superiority count (largest non-zero standard deviation, at alpha divided by the number of non-zero signals), and never fewer than 20. The calculation is `docs/report/power_v3.py`.

Seeds per signal: N_SEEDS = 46, seeds 1000 to 1045, from the `docs/report/power_v3.py` run of 15 November 2026. Pilot standard deviations 0.04904, 0.05539 and 0.05984 at signals 0.0, 0.3 and 0.6 (n = 20 each), inflated to 0.05771, 0.06519 and 0.07043. Equivalence needs 34 seeds; superiority, at alpha 0.05 / 2 with the largest inflated standard deviation, needs 46, which sets the count.

## Endpoints

Balanced accuracy is computed on the out-of-sample block of each seed.

Primary (a), zero signal. Equivalence to chance by two one-sided tests. The 90% interval (1 minus 2 alpha), mean plus or minus the one-sided t critical value times the standard deviation over the square root of n, must lie strictly inside 0.50 plus or minus DELTA.

Primary (b), each non-zero signal. Superiority over 0.50 by a one-sided one-sample t test. P-values are Holm-adjusted across the non-zero signals at ALPHA; a signal is claimed when its adjusted p-value is below ALPHA. The one-sided 95% lower bound is also reported.

Secondary (c). Per signal, the paired gap oracle minus model over the same seeds, with its mean and two-sided 95% t interval.

Descriptive. Counts of seeds where the gate passed, per signal.

Only seeds with a value for a statistic enter it, and the n used is reported.

## Analysis code

`src/aer/demo/control_v3.py` (function `analyse`) implements these rules. The run script is `docs/report/run_control_v3.py`, and results go to `docs/data/detection_curve_v3.csv`.

## Deviations

Any change to a constant, an endpoint, the seed set or the analysis code after the specification hash is published is a deviation and is reported as one, with its reason. A run that fails to complete is rerun unchanged, and both attempts are reported. No endpoint will be added after the run.

## Command

`uv run python docs/report/run_control_v3.py`
