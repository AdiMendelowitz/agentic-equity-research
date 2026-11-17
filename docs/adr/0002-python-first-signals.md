# ADR 0002: Python-first signals

## Status

Accepted

## Context

A model call costs money or free-tier quota, varies between runs and cannot be unit tested at a boundary. Many labels in the pipeline are threshold rules, such as a volatility regime or a signal strength band, and a model adds nothing to them.

## Decision

Every threshold and classification that can be written as code runs as a deterministic function before any model call. The functions live in `src/aer/utils/python_replacements.py` (for example `classify_vix_regime`, `signal_label` and `signal_action_from_classification`) and the indicators are computed in `src/aer/indicators/technical.py`. The `LLM required` comments in `src/aer/agents/` limit the model call to qualitative narrative, and the macro, positioning and sentiment agents note that their label is pre-classified.

## Alternatives considered

- **Let the model classify from raw numbers.** Boundaries become unreproducible and cannot be tested at the exact threshold.
- **Use a model to confirm each rule.** It doubles the calls for a result the code already fixes.
- **Keep thresholds inline in the prompts.** The numbers then drift from the code that scores them and no test can pin them.

## Consequences

- Positive. Boundary behaviour is exact and tested, the offline demo needs no model call, and the model call is limited to narrative.
- Negative. A threshold is a fixed choice that the code encodes, and changing it is a code change with a test change.
- No check yet confirms that an agent declared deterministic imports no model client. The lessons document lists this as planned.

## Enforced by

- `tests/utils/test_python_replacements.py` (for example `test_all_boundaries_are_distinct`)
- `tests/test_signal_boundaries.py` (for example `test_adx_at_threshold_is_trend`, `test_rsi_overbought_boundary`)
