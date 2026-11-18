# ADR 0005: Evaluation unit and pre-registration

## Status

Accepted

## Context

Two questions decide whether a published result can be trusted. The first is what counts as one observation. The second is whether the test could have found a signal that exists, with its rules fixed before the results were read.

## Decision

The unit of every figure on recorded output is one decision per ticker-day, keeping the first run of each day. The pre-registered control counts one observation per seed and signal strength. Public figures first counted same-day reruns as separate calls. Of 2,609 rows first published, 288 were such reruns, and the 78 of them that were resolved calls had been scored against the same outcome as the first run. The correction of 17 November 2026 moved the scored cohort from 342 correct of 707 to 314 of 629, with the old and new figures side by side in the README.

The detection control, version 3, is pre-registered. Its specification is `docs/prereg/control-v3.md`, with constants mirrored in `src/aer/demo/prereg_v3.py`. A hash covering the specification, `prereg_v3.py` and `control_v3.py` was posted in public issue 1 before the first public commit of results. It fixes 46 seeds per signal strength, a margin of 0.03, and an equivalence test at zero signal and Holm-adjusted superiority tests at the other two strengths.

## Alternatives considered

- **Count every stored row.** Reruns are not independent observations.
- **Keep only the last run of each day.** The first run is the earliest decision available. The headline count is 314 of 629 either way, and the resolved ML predictions move from 64 of 120 to 63 of 121.
- **State expectations after the run.** A reader could not tell a test from a fit to the result.

## Consequences

- Positive. A null result and its own correction are published, and the hash posted in issue 1 shows that the specification existed before the public results.
- Negative. Evaluation writes carry no run identifier, so the unit is applied when the aggregates are built. A run identifier is on the roadmap. The start of the run is recorded only in a local log.

## Enforced by

- `tests/test_control_v3.py` (`test_holm_hand_values`, `test_equivalence_true_when_centred_on_chance`, `test_markdown_matches_prereg_constants`, `test_seeds_raise_when_unset_and_are_disjoint_when_set`)
- `tests/test_readme_figures.py` (`test_rule_baselines_follow_from_the_recommendation_counts`, `test_readme_quotes_the_counts_the_hero_figure_draws`)
- `tests/test_evaluation_spec.py`
