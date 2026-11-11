# Lessons from a first system

These notes record what a first system taught, grouped by the six capabilities in this repository and then by cross-cutting practice. Each lesson names the test or check that enforces it. Where no enforcing check exists yet, the entry says Planned and states what it will check.

## Multi-agent pipeline

Keep the graph shape fixed at five parallel agents and one synthesis node, and let a failed agent degrade one section of the brief rather than the whole run.

Enforced by `tests/test_gates.py` (`test_graph_invocation_allowlist`), `tests/test_graph.py` and `tests/test_circuit_breaker.py`.

## Cost-governed routing

Meter every call from the usage the provider reports, count a call without usage rather than estimating it, and keep credentials out of import time.

Enforced by `tests/utils/test_usage_meter.py` (`test_a_call_without_usage_is_counted_not_guessed`), `tests/test_ollama_cascade.py` for the fallback order and `tests/test_hermetic.py` for imports with no credentials.

## Python-first signal logic

Put every threshold and classification that can be written as code into deterministic functions, and call a model only for what code cannot do.

Enforced by `tests/utils/test_python_replacements.py` and `tests/test_signal_boundaries.py`. Planned: a test that each agent declares whether it needs a model and that agents declared deterministic import no LLM client.

## ML layer

Define the label at the forecast horizon, build features only from past bars, purge the cross-validation gap, and let the gate abstain when the model cannot predict.

Enforced by `tests/ml/test_leakage.py` for label alignment, future-bar mutation and purge-gap geometry, and by `tests/test_ml.py` (`test_high_std_returns_skipped`) and `tests/test_control.py` (`test_skip_reason_branches`) for each reason the gate abstains. The control experiment showed the gate abstaining at a similar rate with and without signal, so a conformal gate replaces it in v0.2.

## Evaluation

Freeze the evaluation definition before looking at results, give every output class a resolution rule (a class with no rule, such as HOLD in the first system, silently leaves most rows unscored), and state the power of the test. A result means little until a positive control shows the harness can find a signal that exists and a negative control shows it reports none when there is none.

Enforced by `tests/test_evaluation_spec.py` (`test_every_class_has_a_rule`), `tests/test_power.py` and `tests/test_equity_curve_non_overlapping.py`. The control expectations V2-E0 to V2-E4 each have a met and a not-met test in `tests/test_control.py`, and `tests/test_gates.py` (`test_aggregate_schemas`) pins the headers of both control tables.

## Operations

Make a dead or hung run visible while it happens, and keep the data sources in a declarative registry so that adding one is a single entry.

Enforced by `tests/test_pipeline_heartbeat.py` and `tests/test_prefetch_manifest.py`.

## Cross-cutting practice

Publish aggregates only, behind a schema gate. Scan for redaction before release. Rebuild figures byte-for-byte from the published data. Audit licences in CI. Keep recorded model output away from any code that scores.

Enforced by `tests/test_gates.py` (`test_aggregate_schemas`, `test_redaction_scan_clean`, `test_demo_never_reaches_ml_or_evaluation`), `tests/test_report_figures.py` and the licence audit step in `.github/workflows/ci.yml`.
