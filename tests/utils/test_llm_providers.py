"""Tests for src/utils/llm_providers.py."""

from unittest.mock import MagicMock, patch

import pytest

from aer.utils.llm_providers import call_nvidia_nim, call_openrouter


def test_call_openrouter_raises_on_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import aer.utils.llm_providers as p

    p._openrouter_client = None
    with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
        call_openrouter("m", "sys", "user", 100, False, 10.0)


def test_call_openrouter_raises_on_empty_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = None
    with patch("aer.utils.llm_providers._get_openrouter_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_resp
        with pytest.raises(ValueError, match="Empty response"):
            call_openrouter("m", "sys", "user", 100, False, 10.0)


def test_call_openrouter_sets_json_mode_in_request(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = '{"ok": true}'
    with patch("aer.utils.llm_providers._get_openrouter_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_resp
        call_openrouter("m", "sys", "user", 100, json_mode=True, timeout=10.0)
        call_kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("response_format") == {"type": "json_object"}


def test_call_openrouter_omits_json_mode_when_false(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "plain text"
    with patch("aer.utils.llm_providers._get_openrouter_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_resp
        call_openrouter("m", "sys", "user", 100, json_mode=False, timeout=10.0)
        call_kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
        assert "response_format" not in call_kwargs


def test_call_nvidia_nim_raises_on_missing_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    import aer.utils.llm_providers as p

    p._nim_client = None
    with pytest.raises(EnvironmentError, match="NVIDIA_NIM_API_KEY"):
        call_nvidia_nim("m", "sys", "user", 100, 10.0)


def test_call_nvidia_nim_raises_on_empty_response(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = None
    with patch("aer.utils.llm_providers._get_nim_client") as mock_client:
        mock_client.return_value.chat.completions.create.return_value = mock_resp
        with pytest.raises(ValueError, match="Empty response"):
            call_nvidia_nim("m", "sys", "user", 100, 10.0)


def test_openrouter_client_is_singleton(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    import aer.utils.llm_providers as p

    p._openrouter_client = None
    with patch("openai.OpenAI", return_value=MagicMock()) as mock_ctor:
        from aer.utils.llm_providers import _get_openrouter_client

        c1 = _get_openrouter_client()
        c2 = _get_openrouter_client()
    assert c1 is c2
    assert mock_ctor.call_count == 1
