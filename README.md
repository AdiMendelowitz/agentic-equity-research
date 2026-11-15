# agentic-equity-research

**A multi-agent LLM equity-research pipeline with cost-governed model routing, deterministic signal code, a calibrated ML gate that can abstain, and an evaluation that reports what it can and cannot detect.**

[![CI](https://github.com/AdiMendelowitz/agentic-equity-research/actions/workflows/ci.yml/badge.svg)](https://github.com/AdiMendelowitz/agentic-equity-research/actions/workflows/ci.yml)
[![Python 3.12 to 3.14](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-D7FF64)](https://github.com/astral-sh/ruff)

> [!NOTE]
> A research demonstration, not investment advice and not a trading system. By default it runs offline on synthetic data and makes no model call.

[Results](#results-at-a-glance) · [Quick start](#quick-start) · [How it works](#how-it-works) · [Evaluation](#what-the-evaluation-found) · [Control experiment](#does-the-harness-detect-a-signal-that-exists) · [Tests](#what-the-tests-check) · [Limits](#limits) · [Roadmap](#roadmap)

## Results at a glance

- **No measurable edge on recorded output.** 342 of 707 resolved calls were correct (0.484, about 0.45 to 0.52), and detecting 0.55 against 0.50 would need about 783 independent calls.
- **The harness detects a signal that exists, under a known generator.** In a pre-registered control on synthetic prices, the deployed calibrated ensemble scores balanced accuracy 0.530 and 0.565 at the two non-zero signal strengths, both above 0.50 after a Holm adjustment, and stays within 0.03 of chance at zero signal. A rule that thresholds the true conditional mean is 0.104 and 0.145 ahead.
- **The ML gate barely separates signal from noise.** In that control it passed in 4 of 46 seeds with no signal and 11 of 46 at the strongest, a descriptive count and not a tested endpoint; a conformal gate is the next candidate, to be tested against matched-coverage selection.

<p align="center"><img src="docs/img/detection_curve.svg" alt="Model and oracle balanced accuracy against injected signal strength, with the analytic ceiling and the 0.50 chance line" width="680"></p>

## Quick start

The package needs Python 3.12 to 3.14 (3.14 is the default) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run aer-demo   # five synthetic tickers, offline, no model call
```

<details>
<summary>Sample output of <code>uv run aer-demo</code> (excerpt from a fresh-clone check: banner, two of five briefs, summary)</summary>

```text
SYNTHETIC DATA
No market data and no real company.
Demonstration output of a research pipeline. Not investment advice.
Author: Adi Mendelowitz

Brief: SYN4
  Sessions: 700
  Close: 93.79
  RSI: 33.56
  SMA50: 101.42
  Recent trend: FALLING
  ML gate: predicted, calibrated probability of an up move 0.69

Brief: SYN5
  Sessions: 700
  Close: 84.14
  RSI: 35.98
  SMA50: 88.66
  Recent trend: FALLING
  ML gate: abstained, cross-validation score below the gate threshold
  Cross-validated macro F1: 0.50

Tickers: 5, predicted: 1, elapsed seconds: 13.5
```

</details>

The demo computes indicators, signals and the ML gate locally on synthetic prices and scores its own output. Only the published aggregates in `docs/data` were scored on real output, and no real company appears in the demo.

## How it works

<p align="center"><img src="docs/diagrams/architecture.svg" alt="Architecture: five parallel agents, synthesis and the ML gate, with model calls routed cheapest first" width="860"></p>

Blue boxes are deterministic code, lavender the agents, rose the synthesis model call, green the ML gate and the free providers, sand the paid fallback and terracotta an abstention.

- **Multi-agent pipeline.** Five agents and one synthesis node built with LangGraph, in `src/aer/graph.py` and `src/aer/agents/`.
- **Cost-governed routing.** A four-step provider cascade that reaches the paid model only when the free options fail or a prompt exceeds their context limit, with a ledger that counts calls without usage instead of estimating them, in `src/aer/utils/llm_client.py` and `src/aer/utils/usage_meter.py`.
- **Python-first signals.** Thresholds and classifications run as deterministic code before any model call, in `src/aer/indicators/` and `src/aer/utils/python_replacements.py`.
- **ML layer.** LightGBM and logistic regression with purged time-series cross-validation, Platt calibration and a gate that abstains, in `src/aer/ml/`.
- **Evaluation.** An evaluation spec with a resolution rule for every output class, and power arithmetic, in `src/aer/evaluation/`.
- **Operations.** A prefetch manifest, a run heartbeat and run history, in `src/aer/prefetch/`, `src/aer/utils/pipeline_heartbeat.py` and `docs/data/run_history.csv`, where 22 of 29 scheduled runs completed with exit code 0.

## What the evaluation found

The evaluation found no measurable edge. Of 2,609 briefs across 51 tickers, 707 were resolved against a five-day forward return, and 342 of those were correct, an accuracy on resolved actionable calls of 0.484 (about 0.45 to 0.52 by normal approximation). That interval contains 0.50. None of the 1,766 HOLD briefs received a scored correctness outcome, and 136 other briefs were not resolved, so this figure describes the resolved calls only.

<p align="center"><img src="docs/diagrams/evaluation-funnel.svg" alt="Evaluation cohort: 2,609 briefs, 843 actionable, 707 resolved, 342 correct" width="660"></p>

<p align="center"><img src="docs/img/calibration.svg" alt="Hit rate by conviction bin with intervals, against the 0.50 line" width="640"></p>

Detecting a hit rate of 0.55 against 0.50 with 80% power at a two-sided 5% level needs about 780 independent resolved calls (783 by `src/aer/evaluation/power.py`). The sample has 707, and overlapping five-day windows make the independent count lower. The published evaluation could not distinguish a small edge from none.

<details>
<summary>Results by conviction, recommendation and ML gate (descriptive only)</summary>

By conviction bin, the 0.6 bin scored 135 of 305 (0.443), the 0.7 bin 186 of 349 (0.533) and the 0.8 bin 19 of 48 (0.396). By recommendation, BUY scored 114 of 276 (0.413), SELL 121 of 246 (0.492) and AVOID 107 of 185 (0.578). All 1,766 HOLD rows were never resolved. The ML gate predicted on 169 of 2,374 rows, and 74 of the 145 resolved predictions were correct (0.510).

</details>

## Does the harness detect a signal that exists?

A null result means little unless the harness can find a signal when one is there. The control, in its version 2, injects a signal of strength s into synthetic prices (20 seeds per strength, 1,500 sessions each) and scores balanced accuracy, whose expected value without skill is 0.50 at any class balance. The model is one uncalibrated classifier per seed: the gate's cross-validation picks LightGBM or logistic regression, with logistic regression when the gate abstains. The deployed calibrated ensemble is not what this measures. Model figures cover all 20 seeds.

<p align="center"><img src="docs/diagrams/control-design.svg" alt="Control design: a known injected signal, one classifier per seed, compared with the oracle and the analytic ceiling" width="630"></p>

Five expectations, defined in `src/aer/demo/control.py`, were written down before the run in the private development history, and all five were met. They check the oracle against the ceiling, that the zero-signal interval contains 0.50, that the interval at 0.6 lies above 0.50, that accuracy rises with s, and that the model does not exceed the oracle. They are checks on one run, not five independent confirmations.

| Signal strength s | Model balanced accuracy (95% interval) | Oracle balanced accuracy | Analytic ceiling | Gate predicted |
| --- | --- | --- | --- | --- |
| 0.0 | 0.481 (0.459 to 0.502) | 0.500 | 0.500 | 4 of 20 seeds |
| 0.3 | 0.531 (0.506 to 0.555) | 0.621 | 0.641 | 4 of 20 seeds |
| 0.6 | 0.574 (0.548 to 0.600) | 0.702 | 0.717 | 6 of 20 seeds |

The negative control was met narrowly. At zero signal the upper bound of the interval is only 0.002 above 0.50, and the mean sits below 0.50; this shows no false detection here, not that the model is equivalent to chance. At s = 0.6 the model's margin over chance is 0.074, against 0.202 for the oracle. The gate fired in 20%, 20% and 30% of seeds as the signal grew, so its F1 threshold showed no useful discrimination here.

The first run failed as a design and stays on record in `docs/data/detection_curve_run1.csv`. Its injected signal fed back on itself, which turned the strongest setting into a runaway trend, and raw accuracy tracked the label base rate. Version 2 removed the feedback and switched to balanced accuracy. The command is `uv run python docs/report/run_control.py`.

### Version 3, pre-registered

Version 3 scores the deployed calibrated ensemble on 46 fresh seeds per strength. Its specification, `docs/prereg/control-v3.md`, was released in a public commit and hashed in [issue 1](https://github.com/AdiMendelowitz/agentic-equity-research/issues/1) before the first public commit of the results; the seed count comes from a power calculation for a margin of 0.03.

| Signal strength s | Model balanced accuracy | Pre-registered test | Verdict | Gate passed |
| --- | --- | --- | --- | --- |
| 0.0 | 0.516 | 90% interval 0.510 to 0.522 inside 0.47 to 0.53 | within the 0.03 margin of chance: met | 4 of 46 seeds |
| 0.3 | 0.530 | lower bound 0.519, Holm-adjusted p below 0.001 | above chance: met | 5 of 46 seeds |
| 0.6 | 0.565 | lower bound 0.547, Holm-adjusted p below 0.001 | above chance: met | 11 of 46 seeds |

The oracle stays 0.104 and 0.145 ahead at s = 0.3 and 0.6. At zero signal the interval sits inside the margin but above 0.50. Being within the margin is not evidence that the mean equals 0.50. An exploratory rerun on 300 further seeds, reported with its figures on the report page, is consistent with the excess being a finite-window property of the per-seed statistic, does not identify its cause, and has one control interval that misses its expected value. The command is `uv run python docs/report/run_control_v3.py`.

## What the tests check

- **Leakage.** `tests/ml/test_leakage.py` covers label alignment, future-bar mutation and purge-gap geometry. `tests/test_models_fixes.py` checks the purge gap before the holdout block, that no holdout row reaches any fit, and that the logged Brier score is out of fold.
- **Hermetic imports.** `tests/test_hermetic.py` imports every module with no network, database or credentials.
- **Data gates.** `tests/test_gates.py` checks the published data schemas and keeps the demo package out of ML and evaluation code.
- **Publication checks.** A redaction scan (`tools/redact_check.py`), a secrets scan (gitleaks) and a licence audit that fails on GPL and AGPL licence names in the locked environment run in CI.
- **Pre-registered analysis.** `tests/test_control_v3.py` pins the equivalence, Holm and paired-gap rules on hand-built cases and checks that the specification text states the same constants as the code.
- **Versions.** CI runs the test suite and a blocking offline demo on Python 3.12, 3.13 and 3.14.

## Limits

- No real evaluation observation is published at row level, and the evaluation aggregates carry no ticker, week or date dimension. Synthetic control rows and run dates are published.
- Up to v0.1.2 the ML layer had three leakage defects, fixed in v0.1.3 with a test each. The calibration block followed the training block with no purge gap. The logged Brier score was computed on rows the model was fitted on. The Brier check scored the calibrated ensemble on the rows its calibrator was fitted on. Calibration is now fitted on out-of-fold predictions, and a purged holdout block is used only for evaluation.
- Control version 2 measures one uncalibrated classifier per seed, not the deployed ensemble; version 3 measures the ensemble. In version 3 the model scores slightly above 0.50 on pure noise, within the pre-registered margin. In the exploratory analysis the pooled balanced accuracy is at most 0.009 above 0.50 at the upper end of its 95% interval; that bounds this statistic, not every form of dependence, and the cause is not identified.
- Samples are small, with 707 resolved calls and 145 resolved ML predictions.
- Only the LightGBM member of the ensemble is Platt-calibrated, the logged out-of-fold Brier score belongs to the model chosen on the same folds, and the version 3 seed count was planned from version 2 variability, which the strongest signal slightly exceeded.
- The control's oracle thresholds the true conditional mean at the cost and is not the rule that maximises balanced accuracy; the analytic ceiling is for the zero-cost sign problem.
- The control experiment uses one synthetic signal family; it shows the harness can find that signal, not that real prices carry one.
- The control's expectations were fixed before its results in the private development history; each release here is a single commit, so that order cannot be checked from this repository. That applies to versions 1 and 2. From version 3 the specification is released in a public commit, and its hash posted in a public issue, before the first public commit of the results. The hashes for version 3 cover three files; the code they import is fixed by the release commit, and the start of the run is recorded only in a local log.
- Briefs are scored against a five-day move, while the underlying thesis spans weeks to months.
- The published aggregates were not re-scored with the HOLD rule, which was defined after they were seen.
- `numba` is overridden in `pyproject.toml` so that `pandas-ta` resolves on Python 3.14.
- Live mode is absent.

## Roadmap

Planned next:

- The results report, built today by `docs/report/build_report.py`, published on GitHub Pages.
- Exact baselines for the recorded calls, which need one further whole-sample aggregate of forward-return directions.
- An oracle check with the same target and threshold as the analytic ceiling, and a cause for the unexplained control result in the exploratory zero-signal analysis.
- Dependence-aware intervals for the live evaluation, with the method published.
- A conformal gate tested against matched-coverage selection, and a single-agent against multi-agent comparison.
- Indicators written in this repository in place of `pandas-ta`.

## Author and licence

Built by Adi Mendelowitz ([adimendelowitz.dev](https://adimendelowitz.dev)). To cite this repository, use the "Cite this repository" button, which reads `CITATION.cff`.

MIT licence. This repository is a demonstration and is not investment advice.
