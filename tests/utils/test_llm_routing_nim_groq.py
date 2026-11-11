"""R-A (Adi 2026-09-28): ANALYSIS and SYNTHESIS route NVIDIA NIM first, Groq second, paid Anthropic last."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aer.utils import llm_client, llm_config, llm_providers

# meta/llama-3.3-70b-instruct reached end of life on NIM 2026-08-26 (HTTP 410); nim_smoke_20260928-101608
# returned valid JSON 2 of 2 in 5.4-9.4 s for nvidia/nemotron-3-super-120b-a12b
_NIM_DEFAULT = "nvidia/nemotron-3-super-120b-a12b"
# plan_llm_route_2026-09-28 item 3: ANALYSIS uses the smaller Groq model, SYNTHESIS keeps the larger one
_GROQ_DEFAULT = {"ANALYSIS": "openai/gpt-oss-20b", "SYNTHESIS": "openai/gpt-oss-120b"}


@pytest.fixture
def keys(monkeypatch):
    # a test that imports simulator.daily_runner loads.env into the process, so clear every override
    for tier in ("ANALYSIS", "SYNTHESIS"):
        for level in ("PRIMARY", "SECONDARY", "FALLBACK"):
            monkeypatch.delenv(f"MODEL_{tier}_{level}", raising=False)

    def set_keys(nim: bool, groq: bool) -> dict:
        for var, on in (("NVIDIA_NIM_API_KEY", nim), ("GROQ_API_KEY", groq)):
            if on:
                monkeypatch.setenv(var, "test-key")
            else:
                monkeypatch.delenv(var, raising=False)
        return llm_config.load_tier_config()

    return set_keys


@pytest.mark.parametrize("tier", ["ANALYSIS", "SYNTHESIS"])
def test_nim_first_then_groq_when_both_keys_exist(keys, tier):
    cfg = keys(nim=True, groq=True)[tier]
    assert (cfg.primary_provider, cfg.primary_model) == ("nvidia_nim", _NIM_DEFAULT)
    assert (cfg.secondary_provider, cfg.secondary_model) == ("groq", _GROQ_DEFAULT[tier])


def test_paid_fallbacks_are_unchanged(keys):
    cfg = keys(nim=True, groq=True)
    assert cfg["ANALYSIS"].fallback_model == "claude-haiku-4-5-20251001"
    assert cfg["SYNTHESIS"].fallback_model == "claude-sonnet-4-6"


@pytest.mark.parametrize("tier", ["ANALYSIS", "SYNTHESIS"])
def test_without_the_keys_the_openrouter_route_is_unchanged(keys, tier):
    cfg = keys(nim=False, groq=False)[tier]
    assert cfg.primary_provider == cfg.secondary_provider == "openrouter"
    assert cfg.primary_model == "nvidia/nemotron-3-super-120b-a12b:free"


def test_a_missing_groq_key_keeps_the_openrouter_secondary(keys):
    cfg = keys(nim=True, groq=False)["ANALYSIS"]
    assert (cfg.primary_provider, cfg.secondary_provider) == ("nvidia_nim", "openrouter")
    assert cfg.secondary_model == "google/gemma-4-31b-it:free"


def test_an_explicit_model_variable_still_wins(keys, monkeypatch):
    monkeypatch.setenv("MODEL_ANALYSIS_PRIMARY", "nvidia/some-other-model")
    assert keys(nim=True, groq=True)["ANALYSIS"].primary_model == "nvidia/some-other-model"


def test_call_groq_raises_without_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_providers, "_groq_client", None)
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        llm_providers.call_groq("m", "s", "u", max_tokens=10, json_mode=False, timeout=1.0)


def test_call_groq_uses_the_groq_endpoint_and_json_mode(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm_providers, "_groq_client", None)
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content='{"a": 1}'))]
    with patch("openai.OpenAI", return_value=client) as ctor:
        out = llm_providers.call_groq("openai/gpt-oss-120b", "s", "u", max_tokens=10, json_mode=True, timeout=1.0)
    assert out == '{"a": 1}'
    assert ctor.call_args.kwargs["base_url"] == "https://api.groq.com/openai/v1"
    assert client.chat.completions.create.call_args.kwargs["response_format"] == {"type": "json_object"}
    monkeypatch.setattr(llm_providers, "_groq_client", None)


def test_call_groq_raises_on_an_empty_response(monkeypatch):
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=""))]
    monkeypatch.setattr(llm_providers, "_get_groq_client", lambda: client)
    with pytest.raises(ValueError, match="Empty response from Groq"):
        llm_providers.call_groq("m", "s", "u", max_tokens=10, json_mode=False, timeout=1.0)


def test_the_client_dispatches_groq(monkeypatch):
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(llm_client, "call_groq", fake)
    out = llm_client._call_free_provider(
        provider="groq", model="m", system_prompt="s", user_message="u", max_tokens=5, json_mode=True, timeout=1.0
    )
    assert out == "ok" and seen["json_mode"] is True and seen["model"] == "m"


def test_extraction_secondary_on_nim_is_a_live_model(keys):
    cfg = keys(nim=True, groq=True)["EXTRACTION"]
    assert (cfg.secondary_provider, cfg.secondary_model) == ("nvidia_nim", _NIM_DEFAULT)


@pytest.mark.parametrize(
    "getter, env, attr",
    [
        ("_get_openrouter_client", "OPENROUTER_API_KEY", "_openrouter_client"),
        ("_get_nim_client", "NVIDIA_NIM_API_KEY", "_nim_client"),
        ("_get_groq_client", "GROQ_API_KEY", "_groq_client"),
    ],
)
def test_provider_clients_make_no_hidden_sdk_retries(monkeypatch, getter, env, attr):
    """llm_client owns retries (provider contract); the SDK default of 2 retries tripled every timeout."""
    monkeypatch.setenv(env, "test-key")
    monkeypatch.setattr(llm_providers, attr, None)
    with patch("openai.OpenAI", return_value=MagicMock()) as ctor:
        getattr(llm_providers, getter)()
    assert ctor.call_args.kwargs["max_retries"] == 0
    monkeypatch.setattr(llm_providers, attr, None)
