# Data card

This card describes the files in `docs/data`, in the spirit of Datasheets for Datasets (Gebru et al.). It is a research demonstration and not investment advice.

## Contents of `docs/data`

| File | Rows (excluding header) | Holds |
| --- | --- | --- |
| `eval_sample.csv` | 5 | Sample counts as metric and value pairs: ticker-days, tickers, first and last brief month, resolved rows |
| `eval_recommendation.csv` | 4 | Rows, resolved rows and correct calls for each recommendation |
| `eval_calibration.csv` | 6 | Resolved and correct counts by conviction bin |
| `eval_ml_gate.csv` | 7 | Rows, skipped rows, resolved and correct ML predictions by cross-validation F1 bin |
| `eval_backtest_sharpe.csv` | 6 | Row counts by Sharpe bin |
| `run_history.csv` | 29 | Run date, outcome and exit code of scheduled runs; 22 completed with exit code 0, 2 completed with a non-zero code and 5 left no end line |
| `llm_usage.jsonl` | 33 | One usage record per run, with call and token totals and a breakdown by provider and model |
| `detection_curve_run1.csv` | 15 | The first control run, kept on record as a failed design |
| `detection_curve.csv` | 60 | Control version 2, 20 seeds at each of three signal strengths |
| `detection_curve_v3.csv` | 138 | Pre-registered control version 3, 46 seeds at each of three signal strengths |
| `exploratory_zero_signal_v3.csv` | 14 | Summary of an exploratory diagnostic at zero signal, as metric and value pairs |

The control files carry one row per seed and signal strength. Version 3 records the signal, seed, whether the gate passed, the model used, the cross-validation F1 mean and standard deviation, balanced accuracy of the model and of the oracle, the holdout Brier score of the ensemble, the majority-class accuracy, the label base rate and the number of out-of-sample rows.

## Unit of analysis

The unit is the ticker-day, meaning one brief per ticker and day. The README reports 2,321 ticker-days across 51 tickers. Of these, 749 were actionable calls (BUY, SELL or AVOID) and 629 of those were resolved against a five-day forward return; HOLD days carry no scored outcome.

On 17 November 2026 the evaluation aggregates were corrected. The figures first published on 9 November counted same-day reruns as separate calls. Of the 2,609 rows first published, 288 were second or later runs of a ticker-day already in the store. All figures now count one decision per ticker-day and keep the first run of each day. The scored cohort moved from 342 correct of 707 to 314 of 629.

## Provenance

The `eval_*` files are published aggregates of recorded pipeline output. Recorded calls are scored by direction at zero over five days by `src/aer/evaluation/weekly_scorer.py`. The frozen scoring definition for synthetic demo output is in `src/aer/evaluation/spec.py`, and the published aggregates were not re-scored with its rules.

The control files are synthetic. They are written by scripts in `docs/report`:

- `run_control.py` writes `detection_curve.csv`.
- `run_control_v3.py` writes `detection_curve_v3.csv`.
- `explore_zero_signal_v3.py` with `--write` writes `exploratory_zero_signal_v3.csv`.
- `power_v3.py` reads `detection_curve.csv` for standard deviations and prints the seed count.

Each control injects a known signal into synthetic prices, with 1,500 sessions per seed. Seed counts and arithmetic for detectable effects are in `src/aer/evaluation/power.py` and `docs/report/power_v3.py`.

## Synthetic and real

The control data are synthetic. The evaluation aggregates were scored on real recorded output. The offline demo (`uv run aer-demo`) uses synthetic prices and no real company.

## What is deliberately absent

No real evaluation observation is published at row level. The evaluation aggregates carry no ticker, week or date dimension. Synthetic control rows and run dates are published. See "Limits" in the README.

## Known issues

Open limitations are tracked as public issues with the `known-limitation` label: https://github.com/AdiMendelowitz/agentic-equity-research/issues?q=label%3Aknown-limitation

The README "Limits" section lists further caveats, including small samples (629 resolved calls and 120 resolved ML predictions) and the fact that the recorded ML accuracy and the training target are not the same quantity.

## Licence

The repository is released under the MIT licence (`LICENSE`, copyright Adi Mendelowitz). The repository is a demonstration and is not investment advice.

## Rebuilding

From the repository root:

```bash
uv sync --extra dev
uv run python docs/report/build_figures.py
uv run python docs/report/build_report.py --out site
uv run python docs/report/reproduce.py
```

`build_figures.py` regenerates the figures from `docs/data`, and `build_report.py` computes every number on the report page from a file there, recording an input hash in the footer. `reproduce.py` re-runs the pre-registered control and the figure script and compares their output with the committed files, the figures after normalising the sign of zero and hashed element ids, then prints `RESULT: REPRODUCED` or `RESULT: DIFFERS`.
