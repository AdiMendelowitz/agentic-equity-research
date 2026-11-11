"""Tests for src/utils/llm_client.py."""

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from aer.utils.llm_client import LLMResponse, complete


@pytest.fixture(autouse=True)
def _no_live_ollama():
    """Hermetic by default: a real local daemon must not answer these routing tests.

    test_complete_skips_free_on_large_context passed only while Ollama happened to be down; on
    2026-09-24 the Windows gate ran with the daemon up and qwen2.5:3b answered the SYNTHESIS call,
    so the expected Anthropic fallback never happened. Tests that exercise the Ollama tier patch
    is_ollama_available themselves; their inner patch overrides this one.
    """
    with patch("aer.utils.llm_client.is_ollama_available", return_value=False):
        yield


# ── Routing tests ─────────────────────────────────────────────────────────────


def test_complete_uses_free_provider_on_success(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with patch("aer.utils.llm_client.call_openrouter", return_value='{"ok": true}'):
        result = complete("sys", "user", tier="EXTRACTION", json_mode=True)
    assert result.was_fallback is False
    assert result.provider_used != "anthropic"
    assert result.content == '{"ok": true}'


def test_complete_falls_back_to_anthropic_when_all_free_fail(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_text_block = MagicMock()
    mock_text_block.text = '{"verdict": "HOLD"}'

    mock_anthropic_resp = MagicMock()
    mock_anthropic_resp.content = [mock_text_block]
    mock_anthropic_resp.usage.input_tokens = 100
    mock_anthropic_resp.usage.output_tokens = 10

    with (
        patch("aer.utils.llm_client.call_openrouter", side_effect=Exception("429")),
        patch("aer.utils.llm_client.is_ollama_available", return_value=False),
        patch("aer.utils.llm_client._get_anthropic_client") as mock_client,
    ):
        mock_client.return_value.messages.create.return_value = mock_anthropic_resp
        result = complete("sys", "user", tier="EXTRACTION")
    assert result.was_fallback is True
    assert result.provider_used == "anthropic"


def test_complete_skips_free_on_large_context():
    from aer.utils import llm_client

    large_prompt = "x" * (llm_client.FREE_MODEL_CONTEXT_LIMIT * 4 + 100)

    mock_text_block = MagicMock()
    mock_text_block.text = "result"

    mock_resp = MagicMock()
    mock_resp.content = [mock_text_block]
    mock_resp.usage.input_tokens = 500
    mock_resp.usage.output_tokens = 5

    with patch("aer.utils.llm_client._get_anthropic_client") as mock_client:
        mock_client.return_value.messages.create.return_value = mock_resp
        result = complete(large_prompt, "short", tier="SYNTHESIS")

    assert result.was_fallback is True
    assert any("context_too_large" in entry for entry in result.failure_chain)


def test_complete_raises_runtime_error_when_all_fail(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with (
        patch("aer.utils.llm_client.call_openrouter", side_effect=Exception("fail")),
        patch("aer.utils.llm_client.is_ollama_available", return_value=False),
        patch("aer.utils.llm_client._get_anthropic_client") as mock_client,
    ):
        mock_client.return_value.messages.create.side_effect = Exception("anthropic down")
        with pytest.raises(RuntimeError, match="All providers failed"):
            complete("sys", "user", tier="EXTRACTION")


def test_complete_raises_value_error_on_invalid_tier():
    with pytest.raises(ValueError, match="Invalid tier"):
        complete("sys", "user", tier="NONEXISTENT")


# ── Ollama middle-tier tests ──────────────────────────────────────────────────


def test_ollama_used_when_free_providers_fail(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with (
        patch("aer.utils.llm_client.call_openrouter", side_effect=Exception("fail")),
        patch("aer.utils.llm_client.is_ollama_available", return_value=True),
        patch("aer.utils.llm_client.ollama_call", return_value="ollama result") as mock_ollama,
    ):
        result = complete("sys", "user", tier="ANALYSIS")
    assert result.provider_used == "ollama"
    assert result.was_fallback is False
    assert result.content == "ollama result"
    mock_ollama.assert_called_once()


def test_ollama_unavailable_falls_through_to_anthropic(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_text_block = MagicMock()
    mock_text_block.text = "anthropic result"

    mock_anthropic_resp = MagicMock()
    mock_anthropic_resp.content = [mock_text_block]
    mock_anthropic_resp.usage.input_tokens = 50
    mock_anthropic_resp.usage.output_tokens = 5
    with (
        patch("aer.utils.llm_client.call_openrouter", side_effect=Exception("fail")),
        patch("aer.utils.llm_client.is_ollama_available", return_value=False),
        patch("aer.utils.llm_client._get_anthropic_client") as mock_client,
    ):
        mock_client.return_value.messages.create.return_value = mock_anthropic_resp
        result = complete("sys", "user", tier="ANALYSIS")
    assert result.provider_used == "anthropic"
    assert result.was_fallback is True
    assert any(
        isinstance(entry, dict) and "daemon not running" in entry.get("reason", "") for entry in result.failure_chain
    )


def test_ollama_bad_json_falls_through_to_anthropic(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_text_block = MagicMock()
    mock_text_block.text = '{"ok": true}'

    mock_anthropic_resp = MagicMock()
    mock_anthropic_resp.content = [mock_text_block]
    mock_anthropic_resp.usage.input_tokens = 50
    mock_anthropic_resp.usage.output_tokens = 5
    with (
        patch("aer.utils.llm_client.call_openrouter", side_effect=Exception("fail")),
        patch("aer.utils.llm_client.is_ollama_available", return_value=True),
        patch("aer.utils.llm_client.ollama_call", return_value="not json at all"),
        patch("aer.utils.llm_client._get_anthropic_client") as mock_client,
    ):
        mock_client.return_value.messages.create.return_value = mock_anthropic_resp
        result = complete("sys", "user", tier="EXTRACTION", json_mode=True)
    assert result.provider_used == "anthropic"
    assert any(isinstance(entry, dict) and entry.get("provider") == "ollama" for entry in result.failure_chain)


def test_ollama_uses_fast_model_for_extraction():
    from aer.utils.llm_client import OLLAMA_FAST_MODEL

    with (
        patch("aer.utils.llm_client.call_openrouter", side_effect=Exception("fail")),
        patch("aer.utils.llm_client.is_ollama_available", return_value=True),
        patch("aer.utils.llm_client.ollama_call", return_value='{"result": "ok"}') as mock_ollama,
    ):
        complete("sys", "user", tier="EXTRACTION")
    call_kwargs = mock_ollama.call_args
    assert call_kwargs.kwargs.get("model") == OLLAMA_FAST_MODEL or (
        call_kwargs.args and call_kwargs.args[0] == OLLAMA_FAST_MODEL
    )


# ── JSON retry tests ──────────────────────────────────────────────────────────


def test_json_retry_appends_suffix_on_invalid_json(monkeypatch):
    from aer.utils.llm_client import JSON_RETRY_SUFFIX

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    call_count = 0
    suffix_seen = False

    def mock_openrouter(**kwargs):
        nonlocal call_count, suffix_seen
        call_count += 1
        user_message = kwargs.get("user_message", "")
        if JSON_RETRY_SUFFIX in user_message:
            suffix_seen = True
            return '{"ok": true}'
        return "not json"

    with patch("aer.utils.llm_client.call_openrouter", side_effect=mock_openrouter):
        complete("sys", "user", tier="EXTRACTION", json_mode=True)

    assert suffix_seen, "JSON_RETRY_SUFFIX was not appended on retry"
    assert call_count == 2


# ── LLMResponse immutability tests ───────────────────────────────────────────


def test_llm_response_failure_chain_is_tuple():
    resp = LLMResponse(
        content="x",
        model_used="m",
        provider_used="anthropic",
        input_tokens=None,
        output_tokens=None,
        was_fallback=False,
        failure_chain=("a", "b"),
    )
    assert isinstance(resp.failure_chain, tuple)


def test_llm_response_frozen_raises_on_assignment():
    resp = LLMResponse(
        content="x",
        model_used="m",
        provider_used="anthropic",
        input_tokens=None,
        output_tokens=None,
        was_fallback=False,
        failure_chain=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        resp.content = "y"  # type: ignore[misc]


# ── Token estimation test ─────────────────────────────────────────────────────


def test_estimate_tokens_uses_string_concat_not_int_multiply():
    from aer.utils.llm_client import FREE_MODEL_CONTEXT_LIMIT, _estimate_tokens

    large = "a" * (FREE_MODEL_CONTEXT_LIMIT * 4 + 4)
    assert _estimate_tokens(large) > FREE_MODEL_CONTEXT_LIMIT


# ── Prompt caching tests ──────────────────────────────────────────────────────


def test_anthropic_fallback_applies_cache_control_on_long_prompt():
    from aer.utils.llm_client import CACHE_MIN_TOKENS, CHARS_PER_TOKEN

    long_system = "x" * (CACHE_MIN_TOKENS * CHARS_PER_TOKEN)

    mock_text_block = MagicMock()
    mock_text_block.text = "ok"

    mock_resp = MagicMock()
    mock_resp.content = [mock_text_block]
    mock_resp.usage.input_tokens = 1100
    mock_resp.usage.output_tokens = 1

    with (
        patch("aer.utils.llm_client.call_openrouter", side_effect=Exception("fail")),
        patch("aer.utils.llm_client.is_ollama_available", return_value=False),
        patch("aer.utils.llm_client._get_anthropic_client") as mock_client,
    ):
        mock_client.return_value.messages.create.return_value = mock_resp
        complete(long_system, "user", tier="SYNTHESIS")
        call_kwargs = mock_client.return_value.messages.create.call_args
        system_arg = call_kwargs.kwargs.get("system") or call_kwargs.args[0]
        assert any(
            block.get("cache_control") == {"type": "ephemeral"} for block in system_arg
        ), "cache_control not found in system block for long prompt"


def test_anthropic_fallback_no_cache_control_on_short_prompt():
    mock_text_block = MagicMock()
    mock_text_block.text = "ok"

    mock_resp = MagicMock()
    mock_resp.content = [mock_text_block]
    mock_resp.usage.input_tokens = 10
    mock_resp.usage.output_tokens = 1

    with (
        patch("aer.utils.llm_client.call_openrouter", side_effect=Exception("fail")),
        patch("aer.utils.llm_client.is_ollama_available", return_value=False),
        patch("aer.utils.llm_client._get_anthropic_client") as mock_client,
    ):
        mock_client.return_value.messages.create.return_value = mock_resp
        complete("short system", "user", tier="EXTRACTION")
        call_kwargs = mock_client.return_value.messages.create.call_args
        system_arg = call_kwargs.kwargs.get("system") or call_kwargs.args[0]
        assert not any(
            "cache_control" in block for block in system_arg
        ), "cache_control found in system block for short prompt"


# ── Architectural enforcement ─────────────────────────────────────────────────


def test_no_direct_anthropic_imports_outside_llm_client():
    """No file under src/aer may import anthropic directly except src/aer/utils/llm_client.py."""
    from pathlib import Path

    streaming_exceptions = ["src/aer/utils/llm_client.py"]

    project_root = Path(__file__).parents[2]
    src_path = project_root / "src"
    violations = []
    for py_file in src_path.rglob("*.py"):
        rel_posix = py_file.relative_to(project_root).as_posix()
        if any(exc in rel_posix for exc in streaming_exceptions):
            continue
        text = py_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Line-anchored so the check matches real import statements only, not regex
            # strings or docstrings that mention "import anthropic" (e.g. the final_gate.py
            # guard that greps for that very pattern).
            if stripped.startswith("import anthropic") and "langchain" not in stripped:
                violations.append(f"{rel_posix}: {stripped}")
            elif stripped.startswith("from anthropic") and "import" in stripped:
                violations.append(f"{rel_posix}: {stripped}")

    assert not violations, "Direct anthropic imports found outside allowed files:\n" + "\n".join(violations)


# ── JSON fence stripping (json_mode contract) ─────────────────────────────────


def test_strip_json_fences_unit():
    from aer.utils.llm_client import _strip_json_fences

    assert _strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_json_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_json_fences('  {"a": 1}  ') == '{"a": 1}'
    # No closing fence (truncated): still strips the opening fence, leaving the body.
    assert _strip_json_fences('```json\n{"a": 1}') == '{"a": 1}'


def test_anthropic_fallback_strips_json_fences_when_json_mode(monkeypatch):
    """Haiku via the Anthropic path has no response_format JSON mode and wraps output
    in ```json fences despite instructions. json_mode=True must yield bare JSON."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_text_block = MagicMock()
    mock_text_block.text = '```json\n{"position": "LONG"}\n```'

    mock_resp = MagicMock()
    mock_resp.content = [mock_text_block]
    mock_resp.usage.input_tokens = 100
    mock_resp.usage.output_tokens = 20
    with (
        patch("aer.utils.llm_client.call_openrouter", side_effect=Exception("429")),
        patch("aer.utils.llm_client.is_ollama_available", return_value=False),
        patch("aer.utils.llm_client._get_anthropic_client") as mock_client,
    ):
        mock_client.return_value.messages.create.return_value = mock_resp
        result = complete("sys", "user", tier="EXTRACTION", json_mode=True)
    assert result.provider_used == "anthropic"
    import json

    assert json.loads(result.content) == {"position": "LONG"}
