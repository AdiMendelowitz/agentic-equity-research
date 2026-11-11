# tests/test_ollama_cascade.py
from unittest.mock import patch

import pytest

from aer.utils.ollama_client import OLLAMA_CASCADE, ollama_call_with_cascade


@pytest.fixture(autouse=True)
def _ollama_daemon_up():
    """Make these tests hermetic: ollama_call_with_cascade pre-checks is_ollama_available()
    and raises before reaching the mocked ollama_call when no local daemon is running.
    Force the health check True so the cascade logic itself is what's under test."""
    with patch("aer.utils.ollama_client.is_ollama_available", return_value=True):
        yield


def test_cascade_returns_first_model_response():
    with patch("aer.utils.ollama_client.ollama_call", return_value="ok") as mock:
        result, model_used = ollama_call_with_cascade(prompt="test", system="sys", max_tokens=100)
    assert result == "ok"
    assert model_used == OLLAMA_CASCADE[0]
    mock.assert_called_once_with(prompt="test", system="sys", model=OLLAMA_CASCADE[0], max_tokens=100)


def test_cascade_escalates_to_second_model_on_runtime_error():
    def _side_effect(prompt, system, model, max_tokens):
        if model == OLLAMA_CASCADE[0]:
            raise RuntimeError("Ollama unavailable")
        return "fallback"

    with patch("aer.utils.ollama_client.ollama_call", side_effect=_side_effect):
        result, model_used = ollama_call_with_cascade(prompt="test", system="", max_tokens=100)
    assert result == "fallback"
    assert model_used == OLLAMA_CASCADE[1]


def test_cascade_escalates_on_any_exception():
    """openai.APIError and other non-RuntimeError exceptions must also trigger escalation."""

    class _FakeAPIError(Exception):
        pass

    def _side_effect(prompt, system, model, max_tokens):
        if model == OLLAMA_CASCADE[0]:
            raise _FakeAPIError("connection refused")
        return "fallback"

    with patch("aer.utils.ollama_client.ollama_call", side_effect=_side_effect):
        result, model_used = ollama_call_with_cascade(prompt="test", system="", max_tokens=100)
    assert result == "fallback"
    assert model_used == OLLAMA_CASCADE[1]


def test_cascade_raises_when_all_models_fail():
    with patch("aer.utils.ollama_client.ollama_call", side_effect=RuntimeError("fail")):
        with pytest.raises(RuntimeError, match="exhausted"):
            ollama_call_with_cascade(prompt="test", system="", max_tokens=100)


def test_cascade_accepts_custom_models():
    with patch("aer.utils.ollama_client.ollama_call", return_value="custom"):
        result, model_used = ollama_call_with_cascade(prompt="p", system="", models=["custom-model"], max_tokens=50)
    assert model_used == "custom-model"


@pytest.mark.parametrize(
    "default, override, expected",
    [
        ("qwen2.5:7b", None, "qwen2.5:3b"),
        ("qwen2.5:3b", None, "llama3.2:3b"),
        ("qwen2.5:3b", "qwen2.5:3b", "llama3.2:3b"),
        ("qwen2.5:7b", "phi:latest", "phi:latest"),
    ],
)
def test_cascade_always_has_two_distinct_models(default, override, expected):
    """R4 can set AER_OLLAMA_MODEL=qwen2.5:3b; the fallback must then move off 3b."""
    from aer.utils.ollama_client import _pick_fallback

    assert _pick_fallback(default, override) == expected
    assert len(set(OLLAMA_CASCADE)) == 2
