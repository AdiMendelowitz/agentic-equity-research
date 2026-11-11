"""Per-process LLM token meter: provider-reported usage, no extra API calls (token-aware, follow-up).

Every provider call made through aer.utils.llm_client records its `usage` here: OpenAI-compatible
providers report prompt_tokens/completion_tokens, Anthropic reports input/output and cache tokens.
A call whose response carries no usage is counted in `calls_without_usage` and never estimated, so
the totals are measured actuals only. Nothing is written per call: run_pipeline appends one summary
line per run to `llm_usage.jsonl` in the artifact directory. A daily free-tier budget (Groq, per model,
uncached input plus output) logs one WARNING per process when metered tokens reach USAGE_BUDGET_WARN_FRACTION
of it; the budget is daily and this meter is per process, so the warning is a floor, not a guarantee.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aer.utils.constants import GROQ_FREE_TOKENS_PER_DAY, USAGE_BUDGET_WARN_FRACTION

logger = logging.getLogger(__name__)

_DAILY_TOKEN_BUDGET: dict[str, int] = {"groq": GROQ_FREE_TOKENS_PER_DAY}
_FIELDS = ("calls", "in_tokens", "out_tokens", "cache_read", "cache_write", "calls_without_usage")

_lock = threading.Lock()
_totals: dict[tuple[str, str], dict[str, int]] = {}
_warned: set[str] = set()
_local = threading.local()


def reset() -> None:
    with _lock:
        _totals.clear()
        _warned.clear()
    _local.last = (None, None)


def record(
    provider: str,
    model: str,
    in_tokens: int | None,
    out_tokens: int | None,
    cache_read: int = 0,
    cache_write: int = 0,
) -> None:
    """Add one call. ``None`` token counts mean the provider reported no usage."""
    has_usage = in_tokens is not None or out_tokens is not None
    _local.last = (in_tokens, out_tokens) if has_usage else (None, None)
    with _lock:
        row = _totals.setdefault((provider, model), dict.fromkeys(_FIELDS, 0))
        row["calls"] += 1
        row["in_tokens"] += in_tokens or 0
        row["out_tokens"] += out_tokens or 0
        row["cache_read"] += cache_read or 0
        row["cache_write"] += cache_write or 0
        row["calls_without_usage"] += 0 if has_usage else 1
        budget = _DAILY_TOKEN_BUDGET.get(provider)
        key = f"{provider}/{model}"
        if budget is None or key in _warned:
            return
        # Groq lists its free limits per model and does not count cached input against them (prompt caching docs)
        used = row["in_tokens"] - row["cache_read"] + row["out_tokens"]
        if used >= budget * USAGE_BUDGET_WARN_FRACTION:
            _warned.add(key)
            logger.warning(
                "%s free-tier tokens (uncached input + output) at %d of the %d daily budget in this process; "
                "later calls may fall to the paid fallback",
                key,
                used,
                budget,
            )


def record_openai_usage(provider: str, model: str, response: Any) -> None:
    """Record an OpenAI-compatible response's usage (prompt_tokens / completion_tokens)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        record(provider, model, None, None)
        return
    cached = _int_or_none(getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)) or 0
    record(
        provider,
        model,
        _int_or_none(getattr(usage, "prompt_tokens", None)),
        _int_or_none(getattr(usage, "completion_tokens", None)),
        cache_read=cached,
    )


def _int_or_none(value: Any) -> int | None:
    """Provider payloads vary; only a real integer counts as a measured token figure."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def last_usage() -> tuple[int | None, int | None]:
    """(in, out) of the last call recorded on this thread, or (None, None)."""
    return getattr(_local, "last", (None, None))


def snapshot() -> list[dict[str, Any]]:
    with _lock:
        return [{"provider": p, "model": m, **row} for (p, m), row in sorted(_totals.items())]


def write_run_summary(path: Path, run_id: str) -> dict[str, Any]:
    """Append one JSON line for this run: totals plus one row per provider and model."""
    rows = snapshot()
    summary = {
        "ts": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "totals": {k: sum(r[k] for r in rows) for k in ("calls", "in_tokens", "out_tokens")},
        "by_model": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")
    return summary
