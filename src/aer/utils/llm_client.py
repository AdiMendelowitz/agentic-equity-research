"""
Unified LLM client - the single entry point for all LLM calls in the project.

ARCHITECTURE
  complete() routes each call through four providers in order:
    1. Free primary   (OpenRouter or NIM, configured per tier in llm_config.py)
    2. Free secondary (OpenRouter or NIM, different from primary)
    3. Ollama middle tier (local CPU inference; zero API cost; skipped if offline)
    4. Anthropic fallback (paid; only reached when all above fail)

  Routing is transparent to callers. All callers receive LLMResponse.
  Callers access only LLMResponse.content - no provider-specific parsing.

STREAMING EXCEPTIONS
  This module does not support streaming. Any agent that uses
  client.messages.stream or stream=True must continue importing
  anthropic directly and is listed here:
    (none - no streaming agents in this project)

FAILURE ROUTING (per provider attempt)
  HTTP 429:          sleep(2^attempt seconds), retry up to FREE_MODEL_MAX_RETRIES
  Timeout:           switch to next provider immediately
  Empty response:    switch to next provider immediately
  Invalid JSON:      append JSON-only instruction to user_message, retry ONCE,
                     then switch to next provider
  EnvironmentError:  switch to next provider immediately
  Any other error:   switch to next provider immediately
  All free failed:   try Ollama middle tier (if localhost:11434 responds)
  Ollama failed/off: fall back to Anthropic (with prompt caching if eligible)
  Anthropic failed:  raise RuntimeError with full failure_chain

CONTEXT WINDOW GUARD
  If (len(system_prompt) + len(user_message)) // 4 > FREE_MODEL_CONTEXT_LIMIT,
  skip all free providers and route directly to Anthropic.

PROMPT CACHING (Anthropic fallback only)
  Applied when len(system_prompt) // 4 >= 1024.
  Never applied on free provider calls.

CORRELATED FAILURE NOTE
  ANALYSIS and SYNTHESIS tiers use OpenRouter for both primary and secondary.
  A sustained OpenRouter outage causes both to fail, triggering Anthropic fallback.
  This is expected and acceptable behaviour.
"""

import dataclasses
import json
import logging
import threading
import time
from typing import Any, Final, Literal

import anthropic
import httpx

from aer.utils import usage_meter
from aer.utils.llm_config import (
    ANTHROPIC_FALLBACK_TIMEOUT,
    FREE_MODEL_CONTEXT_LIMIT,
    FREE_MODEL_MAX_RETRIES,
    FREE_MODEL_TIMEOUT,
    NIM_TIMEOUT_SECONDS,
    TIER_CONFIG,
    ProviderName,
)
from aer.utils.llm_providers import call_groq, call_nvidia_nim, call_openrouter
from aer.utils.ollama_client import OLLAMA_DEFAULT_MODEL, OLLAMA_FAST_MODEL, is_ollama_available, ollama_call

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN: Final[int] = 4
CACHE_MIN_TOKENS: Final[int] = 1024
JSON_RETRY_SUFFIX: Final[str] = "\n\nReturn JSON only. No prose. No markdown fences."


def _strip_json_fences(text: str) -> str:
    """Strip a leading/trailing markdown code fence from json_mode output.

    Claude via the Anthropic path has no response_format JSON mode and wraps
    output in ```json ... ``` despite the no-fences instruction, so json.loads
    fails at char 0. Stripping makes json_mode a real contract across every
    provider. Handles a missing closing fence (truncation). No fence -> text
    is returned stripped of surrounding whitespace only.
    """
    s = text.strip()
    if s.startswith("```"):
        newline = s.find("\n")
        s = s[newline + 1 :] if newline != -1 else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


_anthropic_client: anthropic.Anthropic | None = None
_anthropic_client_lock: threading.Lock = threading.Lock()

# Tracks (provider, model) pairs that returned 404/400 - permanent failures.
# Checked at the start of each attempt so dead endpoints are skipped instantly
# rather than wasting an HTTP round-trip per call.
_dead_endpoints: set[tuple[str, str]] = set()
_dead_endpoints_lock: threading.Lock = threading.Lock()


def reset_dead_endpoints() -> None:
    """Clear the dead-endpoint cache. Call in test teardown to restore provider availability."""
    with _dead_endpoints_lock:
        _dead_endpoints.clear()


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        with _anthropic_client_lock:
            if _anthropic_client is None:
                _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    """
    Normalised response from any LLM provider.

    All fields are immutable. failure_chain is a tuple (not list) to
    enforce immutability - frozen=True prevents field reassignment but
    does not prevent mutation of a list stored in a field.

    was_fallback: True if and only if Anthropic (paid) was used. False whenever
                  any free provider - primary or secondary - succeeded. Equivalent
                  to checking provider_used == "anthropic". Use failure_chain to
                  determine whether the primary free provider specifically failed.
    failure_chain: each entry describes one failed provider attempt,
                   in the order they were tried. Non-empty even on secondary success
                   (records the primary failure). Empty only on clean primary success.
    """

    content: str
    model_used: str
    provider_used: ProviderName
    input_tokens: int | None
    output_tokens: int | None
    was_fallback: bool
    failure_chain: tuple[str, ...]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: character count divided by 4."""
    return len(text) // CHARS_PER_TOKEN


def _call_free_provider(
    provider: ProviderName,
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    json_mode: bool,
    timeout: float,
    temperature: float = 1.0,
    reasoning: str | None = None,
) -> str:
    """
    Dispatch to the correct provider adapter.

    For nvidia_nim with json_mode=True: appends JSON-only instruction to
    user_message before calling call_nvidia_nim, because NIM does not
    reliably support response_format=json_object.

    Raises on any failure - caller handles routing.
    """
    if provider == "openrouter":
        return call_openrouter(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
            temperature=temperature,
        )
    if provider == "nvidia_nim":
        msg = (user_message + JSON_RETRY_SUFFIX) if json_mode else user_message
        return call_nvidia_nim(
            model=model,
            system_prompt=system_prompt,
            user_message=msg,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            reasoning=reasoning,
        )
    if provider == "groq":
        return call_groq(
            model=model,
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
            temperature=temperature,
            reasoning=reasoning,
        )
    raise ValueError(f"Unknown provider: {provider!r}")


def _call_anthropic_fallback(
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    temperature: float = 1.0,
    json_mode: bool = False,
) -> tuple[str, int | None, int | None]:
    """
    Call Anthropic SDK. Applies prompt caching when system prompt is large.
    Returns (content, input_tokens, output_tokens).
    Raises on any failure - caller converts to RuntimeError.

    The Anthropic SDK has no response_format JSON mode, so when json_mode is set
    we append the JSON-only suffix and strip markdown fences from the reply - the
    model otherwise wraps valid JSON in ```json fences that break json.loads.
    """
    client = _get_anthropic_client()

    if json_mode:
        user_message = user_message + JSON_RETRY_SUFFIX

    if _estimate_tokens(system_prompt) >= CACHE_MIN_TOKENS:
        system_block: list[dict] = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_block = [{"type": "text", "text": system_prompt}]

    # User content is NOT cached: per-ticker prompts vary too much for cache_control on the user turn
    # to yield meaningful savings - the system prompt cache covers the high-repetition tokens.
    try:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_block,
            "messages": [{"role": "user", "content": user_message}],
            "timeout": ANTHROPIC_FALLBACK_TIMEOUT,
        }
        if temperature != 1.0:
            request["temperature"] = temperature
        try:
            response = client.messages.create(**request)
        except TypeError as e:
            # SDK versions differ on which sampling parameters Messages.create accepts; the call matters more than the knob.
            if "temperature" in str(e) and "temperature" in request:
                request.pop("temperature")
                response = client.messages.create(**request)
            else:
                raise
    except httpx.TimeoutException as e:
        raise RuntimeError(f"Anthropic fallback timeout ({ANTHROPIC_FALLBACK_TIMEOUT}s): {e}") from e

    # metered before any content check: an empty or non-text answer is still billed
    _u = getattr(response, "usage", None)
    usage_meter.record(
        "anthropic",
        model,
        getattr(_u, "input_tokens", None),
        getattr(_u, "output_tokens", None),
        cache_read=getattr(_u, "cache_read_input_tokens", 0) or 0,
        cache_write=getattr(_u, "cache_creation_input_tokens", 0) or 0,
    )

    if not response.content:
        raise ValueError(f"Empty Anthropic response: no content blocks returned by model {model!r}")
    text_parts = [b.text for b in response.content if hasattr(b, "text")]
    if not text_parts:
        block_types = [type(b).__name__ for b in response.content]
        # Log the tool use attempt but don't fail immediately - let retry handling catch this
        logger.warning(
            "Anthropic returned non-text block for model %r: %s. This may indicate model confusion about tools.",
            model,
            block_types,
        )
        raise ValueError(f"No text blocks in Anthropic response from model {model!r}; got block types: {block_types}")
    content = "".join(text_parts)
    if json_mode and content:
        content = _strip_json_fences(content)
    if not content:
        raise ValueError(f"Empty text in Anthropic response from model {model!r}")

    in_tok = getattr(response.usage, "input_tokens", None) or 0
    out_tok = getattr(response.usage, "output_tokens", None) or 0
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    logger.info(
        "Anthropic call | model=%s | input=%d | output=%d | cache_read=%d | cache_write=%d",
        model,
        in_tok,
        out_tok,
        cache_read,
        cache_write,
    )
    # Validate output never exceeds max_tokens (hard error, not warning)
    if out_tok and out_tok > int(max_tokens):
        logger.error(
            "Anthropic response exceeded max_tokens: got %d > limit %d. "
            "Response is likely truncated — this corrupts financial data.",
            out_tok,
            max_tokens,
        )
        raise RuntimeError(
            f"LLM response truncation detected ({out_tok} > {max_tokens}). "
            f"Response may be incomplete. Aborting call to preserve data integrity."
        )

    if out_tok and out_tok >= int(max_tokens * 0.95):
        logger.warning(
            "Output tokens (%d) approaching max_tokens cap (%d) — consider increasing limit",
            out_tok,
            max_tokens,
        )

    return (
        content,
        in_tok if in_tok else None,
        out_tok if out_tok else None,
    )


RATE_LIMIT_MAX_WAIT_S: Final[float] = 60.0


def _rate_limit_cannot_clear_soon(exc: Exception) -> bool:
    """A per-day limit, or a retry-after longer than a run should sleep, will not clear inside this run."""
    text = str(exc).lower()
    if "per day" in text or "(tpd)" in text or "(rpd)" in text:
        return True
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after", 0)) > RATE_LIMIT_MAX_WAIT_S
    except (TypeError, ValueError):
        return False


def _attempt_provider(
    provider: ProviderName,
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    json_mode: bool,
    failure_chain: list[str],
    temperature: float = 1.0,
    reasoning: str | None = None,
) -> str | None:
    """
    Attempt one provider with retry logic for HTTP 429 and one JSON retry.
    Returns content string on success, None on all failures.
    Appends failure reasons to failure_chain in place.

    Retry policy:
      - HTTP 429: retry up to FREE_MODEL_MAX_RETRIES times with exponential backoff
      - Invalid JSON: one additional call with JSON_RETRY_SUFFIX appended, independent
                      of the 429 retry counter so both retries get their full budget
      - All other failures: no retry, return None immediately
      - 404, 410 and 400 "not a valid model": the endpoint is marked dead for the process
      - A 429 that cannot clear soon (retry-after above RATE_LIMIT_MAX_WAIT_S, or a per-day limit): no sleep,
        the endpoint is marked dead for the process
    """
    timeout = NIM_TIMEOUT_SECONDS if provider == "nvidia_nim" else FREE_MODEL_TIMEOUT
    with _dead_endpoints_lock:
        is_dead = (provider, model) in _dead_endpoints
    if is_dead:
        failure_chain.append(f"{provider}/{model}: skipped (endpoint permanently unavailable)")
        return None

    # 429 retry loop - only controls HTTP-level retries
    content: str | None = None
    for attempt in range(FREE_MODEL_MAX_RETRIES):
        try:
            content = _call_free_provider(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=max_tokens,
                json_mode=json_mode,
                timeout=timeout,
                temperature=temperature,
                reasoning=reasoning,
            )
            break  # got a response - proceed to JSON validation

        except OSError as exc:
            failure_chain.append(f"{provider}/{model}: {exc}")
            return None  # missing key - no point retrying

        except Exception as exc:
            reason = f"{provider}/{model} attempt {attempt + 1}: {exc}"
            failure_chain.append(reason)
            logger.warning("Provider attempt failed: %s", reason)

            exc_str = str(exc)
            # 529 = Anthropic "Overloaded" (capacity), retriable like 429.
            is_rate_limit = (
                "429" in exc_str or "529" in exc_str or "rate" in exc_str.lower() or "overloaded" in exc_str.lower()
            )
            is_timeout = "timeout" in exc_str.lower() or "timed out" in exc_str.lower()
            is_permanent = (
                "404" in exc_str or "410" in exc_str or ("400" in exc_str and "not a valid model" in exc_str.lower())
            )
            if is_rate_limit and _rate_limit_cannot_clear_soon(exc):
                is_permanent = True

            if is_permanent:
                with _dead_endpoints_lock:
                    _dead_endpoints.add((provider, model))
                logger.info("Marking %s/%s as permanently unavailable", provider, model)
                return None

            if is_timeout or not is_rate_limit:
                return None  # don't retry timeouts or non-429 errors

            # 429 - backoff and retry
            if attempt < FREE_MODEL_MAX_RETRIES - 1:
                time.sleep(2.0 ** (attempt + 1))

    if content is None:
        return None

    # JSON validation + one independent retry (not counted against the 429 budget)
    if not json_mode:
        return content

    # json_mode contract: strip any markdown fences so callers get bare JSON
    # regardless of whether the provider honored response_format.
    content = _strip_json_fences(content)

    try:
        parsed = json.loads(content)
        if not parsed:  # empty dict - no fields produced
            failure_chain.append(f"{provider}/{model}: JSON produced empty dict — retrying with suffix")
        else:
            return content
    except json.JSONDecodeError:
        failure_chain.append(f"{provider}/{model}: invalid JSON — retrying with suffix")

    try:
        content = _call_free_provider(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            user_message=user_message + JSON_RETRY_SUFFIX,
            max_tokens=max_tokens,
            json_mode=json_mode,
            timeout=timeout,
            temperature=temperature,
            reasoning=reasoning,
        )
    except Exception as exc:
        failure_chain.append(f"{provider}/{model} JSON retry: {exc}")
        return None

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and not any(k.strip() for k in parsed):
            failure_chain.append(
                f"{provider}/{model}: JSON has no valid keys after suffix retry (got {list(parsed)!r})"
            )
            return None
        return content
    except json.JSONDecodeError:
        failure_chain.append(f"{provider}/{model}: invalid JSON after suffix retry")
        return None


def complete(
    system_prompt: str,
    user_message: str,
    tier: Literal["EXTRACTION", "ANALYSIS", "SYNTHESIS"],
    max_tokens: int = 1000,
    json_mode: bool = False,
    temperature: float = 1.0,
) -> LLMResponse:
    """
    Route an LLM call through free providers with Anthropic fallback.

    Args:
        system_prompt: Static system instructions. Long prompts (>= 1024 tokens)
                       receive Anthropic prompt caching on fallback calls.
        user_message:  Variable user-turn content.
        tier:          "EXTRACTION" | "ANALYSIS" | "SYNTHESIS"
        max_tokens:    Maximum tokens in the response.
        json_mode:     If True, enforce JSON-only output. Caller must also
                       include JSON instruction in user_message for reliability
                       across providers that ignore response_format.
        temperature:   Sampling temperature. Use 0.0 for deterministic extraction calls.

    Returns:
        LLMResponse with .content populated. Check .was_fallback and
        .failure_chain for routing diagnostics.

    Raises:
        ValueError:   if tier is not a valid tier name.
        RuntimeError: if all providers including Anthropic fail.
    """
    if tier not in TIER_CONFIG:
        raise ValueError(f"Invalid tier {tier!r}. Valid values: {list(TIER_CONFIG)}")

    # EXTRACTION tier mandates temperature=0 for deterministic JSON output
    if tier == "EXTRACTION" and temperature != 0.0:
        logger.debug(
            "EXTRACTION tier forced temperature=0 (was %.1f); deterministic extraction requires temperature=0",
            temperature,
        )
        temperature = 0.0

    # json_mode=True prevents markdown fencing on Haiku fallback
    if tier == "EXTRACTION":
        json_mode = True

    config = TIER_CONFIG[tier]
    failure_chain: list[str] = []
    # EXTRACTION and ANALYSIS (DOWNGRADE tiers) ask the free reasoning models for a short trace;
    # SYNTHESIS, the protected call, keeps the provider default
    reasoning = "low" if tier in ("EXTRACTION", "ANALYSIS") else None

    # Context window guard
    estimated_tokens = _estimate_tokens(system_prompt + user_message)
    if estimated_tokens > FREE_MODEL_CONTEXT_LIMIT:
        failure_chain.append(f"context_too_large (~{estimated_tokens} tokens > {FREE_MODEL_CONTEXT_LIMIT})")
        logger.info(
            "Skipping free providers for tier=%s: ~%d tokens exceeds limit %d",
            tier,
            estimated_tokens,
            FREE_MODEL_CONTEXT_LIMIT,
        )
    else:
        # Try primary free provider
        content = _attempt_provider(
            provider=config.primary_provider,
            model=config.primary_model,
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            json_mode=json_mode,
            failure_chain=failure_chain,
            temperature=temperature,
            reasoning=reasoning,
        )
        if content is not None:
            logger.debug("Free primary succeeded: %s/%s", config.primary_provider, config.primary_model)
            return LLMResponse(
                content=content,
                model_used=config.primary_model,
                provider_used=config.primary_provider,
                input_tokens=usage_meter.last_usage()[0],
                output_tokens=usage_meter.last_usage()[1],
                was_fallback=False,
                failure_chain=tuple(failure_chain),
            )

        # Try secondary free provider
        content = _attempt_provider(
            provider=config.secondary_provider,
            model=config.secondary_model,
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            json_mode=json_mode,
            failure_chain=failure_chain,
            temperature=temperature,
            reasoning=reasoning,
        )
        if content is not None:
            logger.debug("Free secondary succeeded: %s/%s", config.secondary_provider, config.secondary_model)
            return LLMResponse(
                content=content,
                model_used=config.secondary_model,
                provider_used=config.secondary_provider,
                input_tokens=usage_meter.last_usage()[0],
                output_tokens=usage_meter.last_usage()[1],
                was_fallback=False,
                failure_chain=tuple(failure_chain),
            )

    # Ollama middle tier - local CPU inference; zero API cost; skipped when offline
    # Single O(1) health-check (not per-model); explicit error discrimination
    _ollama_model = OLLAMA_FAST_MODEL if tier == "EXTRACTION" else OLLAMA_DEFAULT_MODEL
    if is_ollama_available():
        try:
            ollama_content = ollama_call(
                prompt=user_message,
                system=system_prompt,
                model=_ollama_model,
                max_tokens=max_tokens,
            )
            if ollama_content:
                if json_mode:
                    ollama_content = _strip_json_fences(ollama_content)
                    try:
                        parsed = json.loads(ollama_content)
                        if isinstance(parsed, dict) and any(k.strip() for k in parsed):
                            logger.debug("Ollama middle tier succeeded: %s", _ollama_model)
                            return LLMResponse(
                                content=ollama_content,
                                model_used=_ollama_model,
                                provider_used="ollama",
                                input_tokens=None,
                                output_tokens=None,
                                was_fallback=False,
                                failure_chain=tuple(failure_chain),
                            )
                        failure_chain.append(
                            {"provider": "ollama", "failure_type": "invalid_json", "reason": "empty or invalid keys"}
                        )
                    except json.JSONDecodeError:
                        failure_chain.append(
                            {"provider": "ollama", "failure_type": "json_decode_error", "reason": "invalid JSON"}
                        )
                else:
                    logger.debug("Ollama middle tier succeeded: %s", _ollama_model)
                    return LLMResponse(
                        content=ollama_content,
                        model_used=_ollama_model,
                        provider_used="ollama",
                        input_tokens=None,
                        output_tokens=None,
                        was_fallback=False,
                        failure_chain=tuple(failure_chain),
                    )
        except RuntimeError as exc:
            # Daemon error
            if "unavailable" in str(exc).lower():
                logger.warning(
                    "Ollama daemon unavailable; cascading to next tier",
                    extra={"failure_type": "ollama_daemon_down"},
                )
                failure_chain.append({"provider": "ollama", "failure_type": "ollama_daemon_down", "reason": str(exc)})
            else:
                # Model error, not daemon error
                logger.warning(
                    "Ollama model error: %s",
                    exc,
                    extra={"failure_type": "ollama_model_error"},
                )
                failure_chain.append({"provider": "ollama", "failure_type": "ollama_model_error", "reason": str(exc)})
        except Exception as exc:
            logger.warning("Ollama cascade failed: %s", exc, extra={"failure_type": "ollama_error"})
            failure_chain.append({"provider": "ollama", "failure_type": "ollama_error", "reason": str(exc)})
    else:
        logger.debug("Ollama not running locally; skipping middle tier", extra={"failure_type": "ollama_unavailable"})
        failure_chain.append(
            {"provider": "ollama", "failure_type": "ollama_unavailable", "reason": "daemon not running"}
        )

    # Anthropic fallback
    logger.info(
        "Using Anthropic fallback for tier=%s model=%s. Failure chain: %s",
        tier,
        config.fallback_model,
        failure_chain,
    )
    try:
        content, input_tok, output_tok = _call_anthropic_fallback(
            model=config.fallback_model,
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
        return LLMResponse(
            content=content,
            model_used=config.fallback_model,
            provider_used="anthropic",
            input_tokens=input_tok,
            output_tokens=output_tok,
            was_fallback=True,
            failure_chain=tuple(failure_chain),
        )
    except Exception as exc:
        raise RuntimeError(
            f"All providers failed for tier={tier!r}. "
            f"Failure chain: {list(failure_chain)}. "
            f"Anthropic error: {exc}"
        ) from exc
