# Architecture decision records

- [ADR 0001](0001-fresh-history-and-gated-release.md): a fresh repository with no inherited history, released as one scanned commit.
- [ADR 0002](0002-python-first-signals.md): thresholds and classifications run as deterministic code before any model call.
- [ADR 0003](0003-cost-governed-model-routing.md): free providers first, a local model next, a paid fallback last, with usage metered.
- [ADR 0004](0004-ml-gate-purged-cv-abstention.md): purged cross-validation, out-of-fold calibration and a gate that abstains.
- [ADR 0005](0005-evaluation-unit-and-pre-registration.md): one decision per ticker-day, and a control pre-registered in public.
