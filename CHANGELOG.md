# Changelog

Package versions of agentic-equity-research. Until version 0.1.5 the package version stayed 0.1.0 while public commit titles carried their own labels (v0.1.1 to v0.1.5 between 11 and 15 November 2026, then v0.1); this file starts with the first release whose package version, `CITATION.cff` and `uv.lock` agree, and the public commit history is the record before it. Two public commits of 15 November 2026 (5d7635d and 962d6db) are titled v0.1.5; they are not this release.

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
