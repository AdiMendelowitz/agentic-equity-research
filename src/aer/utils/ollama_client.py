"""
Ollama client for the middle tier - tasks that need language judgment but
do not require Anthropic-tier models.

Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1.
All calls go through this module. Callers never import openai directly for
Ollama calls.

Availability: call is_ollama_available() before routing. If unavailable,
callers fall back to their Anthropic equivalent via llm_client.complete().
"""

import logging
import os
import threading
import time
from typing import Any

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
OLLAMA_HEALTHCHECK: str = "http://localhost:11434"

# Recommended models for CPU-only inference (no GPU)
# Pull via: ollama pull <model>
# Model names come from the environment so the RAM-driven switch between
# qwen2.5:7b (4.7 GB) and qwen2.5:3b (1.9 GB) is a configuration change, not a code change.
OLLAMA_DEFAULT_MODEL: str = os.environ.get("AER_OLLAMA_MODEL") or "qwen2.5:7b"  # balanced accuracy
OLLAMA_FAST_MODEL: str = os.environ.get("AER_OLLAMA_FAST_MODEL") or "llama3.2:3b"  # binary decisions only
_FALLBACK_CANDIDATES: tuple[str, ...] = ("qwen2.5:3b", "llama3.2:3b")


def _pick_fallback(default: str, override: str | None) -> str:
    """Second cascade rung: the override, else the first candidate that differs from the default.

    The cascade must hold two distinct models: with AER_OLLAMA_MODEL=qwen2.5:3b (on a
    RAM-short machine) a fixed qwen2.5:3b fallback collapsed it to one rung, so a failure of the
    only model skipped straight to the paid tier.
    """
    if override and override != default:
        return override
    return next(m for m in _FALLBACK_CANDIDATES if m != default)


OLLAMA_FALLBACK_MODEL: str = _pick_fallback(OLLAMA_DEFAULT_MODEL, os.environ.get("AER_OLLAMA_FALLBACK_MODEL"))


# A refused health check cost 4.0 s on some hosts and ran on every cascade that reached Ollama.
# A failure is remembered for this long; a success never is.
OLLAMA_DOWN_CACHE_SECONDS: float = 300.0
_down_until: float = 0.0
_down_lock = threading.Lock()


def reset_ollama_health_cache() -> None:
    global _down_until
    with _down_lock:
        _down_until = 0.0


def is_ollama_available() -> bool:
    """Health check with a negative cache: a recent failure returns False without a request."""
    global _down_until
    with _down_lock:
        if time.monotonic() < _down_until:
            return False
    up = _check_ollama()
    if not up:
        with _down_lock:
            _down_until = time.monotonic() + OLLAMA_DOWN_CACHE_SECONDS
    return up


def _check_ollama() -> bool:
    """Health-check Ollama before routing any calls.

    Single O(1) health-check (not per-model); explicit exception handling
    for daemon-down vs timeout vs request errors.
    Returns False (not raise) so caller can decide next action.
    """
    try:
        r = requests.get(OLLAMA_HEALTHCHECK, timeout=2)
        if r.status_code != 200:
            logger.warning(
                "Ollama health check returned non-200",
                extra={"failure_type": "bad_status", "status_code": r.status_code},
            )
            return False
        return True
    except (requests.exceptions.ConnectionError, ConnectionError):
        # Daemon is down / not accepting connections
        logger.warning("Ollama connection refused", extra={"failure_type": "connection_error"})
        return False
    except (requests.exceptions.Timeout, TimeoutError):
        # Health-check timed out (daemon slow or stuck)
        logger.warning("Ollama health check timeout", extra={"failure_type": "timeout"})
        return False
    except requests.exceptions.RequestException as e:
        # Other request errors (DNS, SSL, etc.)
        logger.warning("Ollama health check failed: %s", e, extra={"failure_type": "request_error"})
        return False


def ollama_call(
    prompt: str,
    system: str = "",
    model: str = OLLAMA_DEFAULT_MODEL,
    json_schema: dict[str, Any] | None = None,
    max_tokens: int = 300,
) -> str:
    """
    Single-turn Ollama call via the OpenAI-compatible endpoint.

    Args:
        prompt:      User-turn content.
        system:      System prompt (optional). Keep under 500 tokens for fast CPU inference.
        model:       Local model name. Use OLLAMA_FAST_MODEL for binary decisions.
        json_schema: If provided, passed as `format=` via the native `ollama` library to
                     enforce structured output at the model level - eliminates parse failures.
                     Caller should json.loads() the returned string.
        max_tokens:  Always set explicitly; never leave unbounded.

    Returns:
        Response content as a string.

    Raises:
        RuntimeError: if Ollama is unavailable - caller must handle fallback.
    """
    if not is_ollama_available():
        raise RuntimeError("Ollama unavailable — caller should use fallback")

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if json_schema is not None:
        # Native Ollama structured output - enforced at model level via format=
        # Must use the native ollama library; OpenAI-compat client does not expose format=
        import ollama as _ollama  # lazy import - only needed when json_schema provided

        response = _ollama.chat(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            format=json_schema,
            options={"temperature": 0, "num_predict": max_tokens},
        )
        content: str = response.message.content or ""
    else:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        completion = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=0,
        )
        content = completion.choices[0].message.content or ""

    logger.info(
        "Ollama call | model=%s | prompt_chars=%d | response_chars=%d",
        model,
        len(prompt),
        len(content),
    )
    return content


# Second rung: a smaller general model that follows OLLAMA_DEFAULT_MODEL. A coder model was
# dropped because it was not installed locally, so the cascade's second step could only fail.
OLLAMA_CASCADE: list[str] = [OLLAMA_DEFAULT_MODEL, OLLAMA_FALLBACK_MODEL]
# LLM required: insight extraction is irreducibly qualitative - financial themes from
#               social or layoff prose cannot be enumerated or pattern-matched in Python


def ollama_call_with_cascade(
    prompt: str,
    system: str = "",
    models: list[str] | None = None,
    max_tokens: int = 300,
) -> tuple[str, str]:
    """
    Try Ollama models in cascade order, escalating on any exception (RuntimeError, openai.APIError, etc).

    Returns (response_text, model_name) for the first model that succeeds.
    Raises RuntimeError if all cascade models fail - caller handles Anthropic fallback.
    """
    if models is None:
        models = list(OLLAMA_CASCADE)
    # The cascade models share ONE daemon: a single health-check
    # detects daemon-down in O(1). Without this, every model in the cascade pays its own
    # failed health-check round-trip before the loop can give up. Distinct failure_type
    # for daemon-down vs per-model error.
    if not is_ollama_available():
        logger.warning("Ollama daemon unavailable — skipping cascade", extra={"failure_type": "ollama_daemon_down"})
        raise RuntimeError("Ollama daemon unavailable — caller should use fallback")
    last_exc: Exception | None = None
    for model in models:
        try:
            response = ollama_call(prompt=prompt, system=system, model=model, max_tokens=max_tokens)
            logger.info("Ollama cascade success: model=%s", model)
            return response, model
        except Exception as exc:
            logger.warning("Ollama cascade: model=%s failed: %s — escalating", model, type(exc).__name__)
            last_exc = exc
    raise RuntimeError(f"Ollama cascade exhausted ({models}): {last_exc}")
