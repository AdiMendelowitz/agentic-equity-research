# Changelog

Package versions of agentic-equity-research. Until version 0.1.5 the package version stayed 0.1.0 while public commit titles carried their own labels (v0.1.1 to v0.1.5 between 11 and 15 November 2026, then v0.1); this file starts with the first release whose package version, `CITATION.cff` and `uv.lock` agree, and the public commit history is the record before it. Two public commits of 15 November 2026 (5d7635d and 962d6db) are titled v0.1.5; they are not this release.

## 0.1.6 (24 November 2026)

Changed
- Dependency updates in `uv.lock` to langgraph 1.2.23, pydantic-settings 2.16.3, numpy 2.5.5, pandas-market-calendars 5.6.1, empyrical-reloaded 0.7.3, anthropic 2.10.0, openai 3.55.1, pytest-asyncio 1.5.0 and ruff 0.17.5, with langgraph-prebuilt 1.1.3 as langgraph 1.2.23 resolves it. The committed figures and the report page rebuild unchanged.
- The Pages workflow uses version 5 of `actions/upload-pages-artifact` and `actions/deploy-pages`.
- `tools/release.ps1` names the package version in the title of each public commit.
- Version 0.1.6 in `pyproject.toml`, `CITATION.cff` and `uv.lock`.

Added
- `tests/test_data_pins.py` pins the content of every file under `docs/data` by sha256, with line endings normalised as git stores them.

Previous public commit: e038c47 (22 November 2026).

## Documentation update to 0.1.5 (22 November 2026, package version unchanged)

Documented
- `docs/cards/data-card.md` states which model's mean cross-validation F1 places each row of `eval_ml_gate.csv` in its bin.
- `docs/prereg/control-v3-deviations.md` states how far issue #33 reaches into `eval_ml_gate.csv`, measured for all but at most 36 of its 2,112 rows, in place of the sentence that called it not measured; the data file is unchanged.
- `docs/cards/model-card.md` and `README.md` say that ML predictions recorded before the two-model ensemble was added carry the selected model's direction, and those recorded with it the LightGBM member's.

## 0.1.5 (22 November 2026)

Fixed
- Cross-validation F1 scores both classes in every fold (issue #33): `f1_score` is called with `labels=[0, 1]` and `zero_division=0`, so a fold whose true labels and predictions are all one class scores 0.5 instead of 1.0. The cross-validation result also counts such folds.
- Walk-forward validation fails the fold when signal generation returns no signal (issue #34), as it already did when signal generation raised, instead of recording a flat position.

Corrected
- One row of `docs/data/detection_curve_v3.csv` (signal 0.6, seed 1008) in three cells, with no change to any verdict, table, committed figure or README number; the report footer shows the new input hash. The deviation is recorded in `docs/prereg/control-v3-deviations.md`; the specification `docs/prereg/control-v3.md` is unchanged.

Documented
- `docs/cards/data-card.md` and `docs/prereg/control-v3-deviations.md`: the control version 2 pilot (`detection_curve.csv`, 10 November 2026) reproduces from the code that produced it, with no fold affected by issue #33, but not from the 0.1.5 code; the provenance of `eval_backtest_sharpe.csv`.

Changed
- Version 0.1.5 in `pyproject.toml`, `CITATION.cff` and `uv.lock`, with a test that the three agree.

Previous public commit: 6093484 (19 November 2026).
