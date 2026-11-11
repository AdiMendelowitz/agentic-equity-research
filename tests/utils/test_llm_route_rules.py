"""plan_llm_route_2026-09-28: reasoning levels, per-provider timeout, 410 and exhausted-429 handling,
Ollama negative cache, cached-token budgets. Each test names the plan item it proves."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests.exceptions

from aer.utils import llm_client, llm_config, llm_providers, ollama_client, usage_meter
from aer.utils.constants import GROQ_FREE_TOKENS_PER_DAY, USAGE_BUDGET_WARN_FRACTION


def _resp(content="ok", **usage):
    u = SimpleNamespace(**usage) if usage else None
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=u)


@pytest.fixture(autouse=True)
def _clean():
    usage_meter.reset()
    llm_client.reset_dead_endpoints()
    yield
    usage_meter.reset()
    llm_client.reset_dead_endpoints()


# -- reasoning level ------------------------------------------------------------------------------
def test_nim_low_reasoning_sends_low_effort_thinking(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = _resp()
    monkeypatch.setattr(llm_providers, "_get_nim_client", lambda: client)
    llm_providers.call_nvidia_nim("m", "s", "u", max_tokens=5, timeout=1.0, reasoning="low")
    body = client.chat.completions.create.call_args.kwargs["extra_body"]
    assert body == {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}


def test_nim_default_reasoning_sends_no_extra_body(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = _resp()
    monkeypatch.setattr(llm_providers, "_get_nim_client", lambda: client)
    llm_providers.call_nvidia_nim("m", "s", "u", max_tokens=5, timeout=1.0)
    assert "extra_body" not in client.chat.completions.create.call_args.kwargs


def test_groq_low_reasoning_sets_reasoning_effort(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value = _resp()
    monkeypatch.setattr(llm_providers, "_get_groq_client", lambda: client)
    llm_providers.call_groq("m", "s", "u", max_tokens=5, json_mode=False, timeout=1.0, reasoning="low")
    assert client.chat.completions.create.call_args.kwargs["reasoning_effort"] == "low"
    llm_providers.call_groq("m", "s", "u", max_tokens=5, json_mode=False, timeout=1.0)
    assert "reasoning_effort" not in client.chat.completions.create.call_args.kwargs


def _capture_free_calls(monkeypatch):
    seen = []

    def fake(**kwargs):
        seen.append(kwargs)
        return '{"a": 1}'

    monkeypatch.setattr(llm_client, "_call_free_provider", fake)
    return seen


@pytest.mark.parametrize("tier, expected", [("ANALYSIS", "low"), ("EXTRACTION", "low"), ("SYNTHESIS", None)])
def test_tier_decides_the_reasoning_level_and_synthesis_is_untouched(monkeypatch, tier, expected):
    seen = _capture_free_calls(monkeypatch)
    llm_client.complete("s", "u", tier=tier, max_tokens=50, json_mode=True)
    assert seen[0]["reasoning"] == expected


# -- per-provider timeout ------------------------------------------------------------------------
def test_nim_gets_its_own_longer_timeout(monkeypatch):
    seen = _capture_free_calls(monkeypatch)
    for provider in ("nvidia_nim", "groq", "openrouter"):
        llm_client._attempt_provider(
            provider=provider,
            model=f"m-{provider}",
            system_prompt="s",
            user_message="u",
            max_tokens=5,
            json_mode=False,
            failure_chain=[],
        )
    timeouts = {c["provider"]: c["timeout"] for c in seen}
    assert timeouts["nvidia_nim"] == llm_config.NIM_TIMEOUT_SECONDS == 60.0
    assert timeouts["groq"] == timeouts["openrouter"] == llm_config.FREE_MODEL_TIMEOUT


# -- 410 is permanent ----------------------------------------------------------------------------
def test_http_410_marks_the_endpoint_dead_for_the_process(monkeypatch):
    calls = []

    def gone(**kwargs):
        calls.append(1)
        raise RuntimeError("Error code: 410 - model has reached its end of life")

    monkeypatch.setattr(llm_client, "_call_free_provider", gone)
    args = dict(provider="nvidia_nim", model="old", system_prompt="s", user_message="u", max_tokens=5, json_mode=False)
    assert llm_client._attempt_provider(**args, failure_chain=[]) is None
    chain: list = []
    assert llm_client._attempt_provider(**args, failure_chain=chain) is None
    assert len(calls) == 1 and "skipped" in chain[0]


# -- an exhausted daily budget is not retried ----------------------------------------------
@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("Error code: 429 - Rate limit reached for model on tokens per day (TPD): Limit 200000"),
        type("E", (Exception,), {})("Error code: 429 - rate limited"),
    ],
)
def test_a_429_that_cannot_clear_soon_is_not_slept_on(monkeypatch, exc):
    if "per day" not in str(exc):
        exc.response = SimpleNamespace(headers={"retry-after": "3600"})
    sleeps, calls = [], []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleeps.append(s))

    def boom(**kwargs):
        calls.append(1)
        raise exc

    monkeypatch.setattr(llm_client, "_call_free_provider", boom)
    args = dict(provider="groq", model="g", system_prompt="s", user_message="u", max_tokens=5, json_mode=False)
    assert llm_client._attempt_provider(**args, failure_chain=[]) is None
    assert sleeps == [] and len(calls) == 1
    chain: list = []
    llm_client._attempt_provider(**args, failure_chain=chain)
    assert len(calls) == 1 and "skipped" in chain[0]


def test_a_short_429_is_still_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        llm_client, "_call_free_provider", MagicMock(side_effect=[RuntimeError("Error code: 429 - slow down"), "x"])
    )
    out = llm_client._attempt_provider(
        provider="groq", model="g", system_prompt="s", user_message="u", max_tokens=5, json_mode=False, failure_chain=[]
    )
    assert out == "x" and sleeps == [2.0]


# -- Ollama negative cache -----------------------------------------------------------------------
def test_a_failed_ollama_health_check_is_cached_and_a_success_is_not(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(ollama_client.time, "monotonic", lambda: now[0])
    get = MagicMock(side_effect=requests.exceptions.ConnectionError("refused"))
    monkeypatch.setattr(ollama_client.requests, "get", get)
    assert ollama_client.is_ollama_available() is False
    assert ollama_client.is_ollama_available() is False
    assert get.call_count == 1
    now[0] += ollama_client.OLLAMA_DOWN_CACHE_SECONDS + 1
    get.side_effect = None
    get.return_value = SimpleNamespace(status_code=200)
    assert ollama_client.is_ollama_available() is True
    assert ollama_client.is_ollama_available() is True
    assert get.call_count == 3


# -- cached tokens and per-model budgets --------------------------------------------
def test_cached_input_is_recorded_from_prompt_tokens_details():
    r = _resp(prompt_tokens=1000, completion_tokens=100, prompt_tokens_details=SimpleNamespace(cached_tokens=800))
    usage_meter.record_openai_usage("groq", "m", r)
    (row,) = usage_meter.snapshot()
    assert (row["in_tokens"], row["cache_read"]) == (1000, 800)


def test_budget_counts_uncached_tokens_per_model(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="aer.utils.usage_meter")
    threshold = int(GROQ_FREE_TOKENS_PER_DAY * USAGE_BUDGET_WARN_FRACTION)
    usage_meter.record("groq", "a", threshold, 0, cache_read=threshold)  # all cached: counts 0
    usage_meter.record("groq", "b", threshold - 1, 0)  # a different model: its own budget
    assert not caplog.records
    usage_meter.record("groq", "b", 1, 0)
    assert len(caplog.records) == 1 and "groq/b" in caplog.records[0].getMessage()


def test_groq_analysis_secondary_defaults_to_the_smaller_model(monkeypatch):
    for tier in ("ANALYSIS", "SYNTHESIS"):
        for level in ("PRIMARY", "SECONDARY", "FALLBACK"):
            monkeypatch.delenv(f"MODEL_{tier}_{level}", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    cfg = llm_config.load_tier_config()
    assert cfg["ANALYSIS"].secondary_model == "openai/gpt-oss-20b"
    assert cfg["SYNTHESIS"].secondary_model == "openai/gpt-oss-120b"
