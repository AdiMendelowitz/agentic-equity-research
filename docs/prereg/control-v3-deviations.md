# Deviations from the control version 3 specification

`docs/prereg/control-v3.md` is unchanged and frozen. It still matches the sha256 posted in public issue 1. This file records, by date, every change made after the release of the control version 3 results. It does not amend the specification.

## 22 November 2026 (package version 0.1.5): single-class fold in the cross-validation F1

### What changed and why

Public issue #33 reported that the per-fold cross-validation F1 in `src/aer/ml/models.py` was computed with `f1_score(average="macro")` and with the labels inferred from each fold. A fold whose true labels and predictions are all one class therefore scored 1.0. Package version 0.1.5 fixes the call to `labels=[0, 1]` and `zero_division=0`, which scores such a fold 0.5.

### Which published rows are affected

The published control version 3 has 138 cells, with 690 logistic folds and 690 LightGBM folds. Exactly one fold was single-class in both truth and predictions: the logistic model at signal 0.6, seed 1008. LightGBM had one fold that was single-class in truth only, and the fix does not change its score.

`docs/data/detection_curve_v3.csv` was corrected in one row, signal 0.6, seed 1008, in three cells:

| Column | Published | Corrected |
| --- | --- | --- |
| `cv_f1_mean` | 0.665139 | 0.565139 |
| `cv_f1_std` | 0.226292 | 0.155679 |
| `holdout_brier_ensemble` | 0.410142 | 0.391249 |

Every other column of that row is unchanged: `gate_passed` is False, `model_used` is logistic, and the balanced accuracies, `majority_accuracy`, `label_base_rate` and `n_oos` are as published. Every other row is unchanged.

### Why the Brier score moves

The ensemble weight is the LightGBM share of the sum of the two models' mean cross-validation F1 (`FittedEnsemble.weight_lgbm` in `src/aer/ml/models.py`). The logistic mean fell by 0.100000 in this row, so the weight changed and the soft vote of the two probabilities changed with it. The ensemble balanced accuracy thresholds the same soft vote, so it could have moved; the replay of 21 November 2026 measured it unchanged at 0.500000. The oracle balanced accuracy does not depend on the model. The selected model (logistic) and the gate verdict are unchanged in this row.

### Effect on the pre-registered analysis

The output of the pre-registered analysis changes in 0 lines. No verdict, table or README number changes and no committed figure changes; the report footer shows a new input hash for `detection_curve_v3.csv`. The pre-registered endpoints use the balanced accuracy of the ensemble, which the replay measured unchanged in every cell.

### Scope checks

- The exploratory zero-signal run (300 seeds, 3,000 folds) reproduces and contains no single-class fold.
- The `aer-demo` excerpt in the README reproduces and contains no single-class fold.
- `docs/data/detection_curve.csv` (control version 2 pilot, run on 10 November 2026) reproduces byte for byte from the development code of that day, and all 600 of its cross-validation fold scores (60 cells, two models, five folds) were checked: none was single-class, so issue #33 did not affect it. It does not reproduce from the current code (see the next entry). `docs/report/power_v3.py` reads the committed pilot file, so the pre-registered seed count is unaffected.
- `docs/data/detection_curve_run1.csv` was produced by code that was never committed, so it cannot be replayed and its exposure to issue #33 is not measured. It is kept as the record of a failed design and no published result rests on it.
- `docs/data/eval_ml_gate.csv` is produced from recorded briefs by code outside this repository, which scored its cross-validation folds with the issue #33 defect. At most 36 of its 2,112 rows could change bin, skip status or ML outcome counts under the fix, and the file is published unchanged. This was checked against the stored F1 values and the pipeline's own logs. Among the 1,973 skipped rows, at most 3 could hold a single-class fold, all in the 0.50 bin. Predictions recorded before the two-model ensemble was added store the F1 of the selected model, and none of them can hold a single-class fold. Of the 139 predictions, 106 match a logged line of the selected model on ticker and date (13 of them one day off, and some from a simulation log whose runs are not separated from live ones), and none of those can hold one. The other 33 have no matching log line and are not measured. A model with one single-class fold in five needs a mean F1 of at least 0.70 to stay within the gate's limit of 0.15 on the standard deviation, and no logged selected mean exceeds 0.604, but for those 33 this is a bound. The 36 rows are the 3 skipped rows and these 33, and the count holds if each of the 106 matches is the run that recorded its prediction. Model fits do not depend on the scoring, so a changed F1 can alter a recorded prediction only by changing which model the gate checks or whether it passes, and the 36 rows include every row where that is possible.

## 22 November 2026: version 2 reproduces only from the code that produced it

The specification states: "Version 2 stays unchanged and reproducible." The committed `docs/data/detection_curve.csv` is unchanged and still reproduces byte for byte from the development code of 10 November 2026 that produced it (replay of 21 November 2026). Replaying `docs/report/run_control.py` with the current code gives different values in all 60 rows. The code it depends on changed afterwards; among other changes, the leakage fixes of 15 November 2026 (release v0.1.3) set aside a 15 per cent holdout block before cross-validation and replaced `cross_val_score` with folds scored one by one. The single change that altered the output was not isolated. The version 2 file is kept as published, and `docs/report/power_v3.py` reads that committed file, so the version 3 seed count does not depend on a rerun.
