# ADR 0003: Cost-governed model routing

## Status

Accepted

## Context

The pipeline makes many model calls per run. Free providers carry rate limits and limited context windows, and a paid model is reliable but costs money on every call.

## Decision

`src/aer/utils/llm_client.py` routes each call through a fixed cascade. It tries a primary free provider, then a secondary free provider, then a local Ollama model when the daemon is running, and reaches the paid Anthropic model only when these fail. A call with an unknown tier raises `ValueError`. A context-window guard estimates the prompt size from its character count and, above `FREE_MODEL_CONTEXT_LIMIT` (32,000 tokens by default), skips the free providers and records `context_too_large` in the failure chain. `src/aer/utils/usage_meter.py` records the usage each provider reports. A call whose response carries no usage is counted in `calls_without_usage` and never estimated. Of 7,033 logged calls over 33 runs, 1,238 (18%) were served by the paid fallback.

## Alternatives considered

- **Paid model for every call.** It costs the most and gives no gain on mechanical extraction.
- **One free provider.** An outage or a rate limit stops the run.
- **Estimate tokens when usage is missing.** Totals would mix measured and guessed values.

## Consequences

- Positive. The paid model is a measured exception, and the usage records show which provider served each call.
- Negative. Free tiers add latency and retries. The meter is per process, so its daily-budget warning is a floor. The context guard uses a rough character estimate.

## Enforced by

- `tests/utils/test_llm_client.py` (`test_complete_falls_back_to_anthropic_when_all_free_fail`, `test_complete_skips_free_on_large_context`, `test_ollama_used_when_free_providers_fail`, `test_ollama_unavailable_falls_through_to_anthropic`, `test_complete_raises_value_error_on_invalid_tier`)
- `tests/utils/test_usage_meter.py` (`test_a_call_without_usage_is_counted_not_guessed`)
- `tests/test_hermetic.py` (imports with no credentials)
