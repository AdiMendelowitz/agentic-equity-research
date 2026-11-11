"""Tests for src/utils/ollama_client.py."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestIsOllamaAvailable:
    def test_returns_true_on_200(self) -> None:
        from aer.utils.ollama_client import is_ollama_available

        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("aer.utils.ollama_client.requests.get", return_value=mock_response):
            assert is_ollama_available() is True

    def test_returns_false_on_non_200(self) -> None:
        from aer.utils.ollama_client import is_ollama_available

        mock_response = MagicMock()
        mock_response.status_code = 503
        with patch("aer.utils.ollama_client.requests.get", return_value=mock_response):
            assert is_ollama_available() is False

    def test_returns_false_on_connection_error(self) -> None:
        from aer.utils.ollama_client import is_ollama_available

        with patch("aer.utils.ollama_client.requests.get", side_effect=ConnectionError("refused")):
            assert is_ollama_available() is False

    def test_returns_false_on_timeout(self) -> None:
        from aer.utils.ollama_client import is_ollama_available

        with patch("aer.utils.ollama_client.requests.get", side_effect=TimeoutError("timed out")):
            assert is_ollama_available() is False

    def test_return_value_is_bool(self) -> None:
        from aer.utils.ollama_client import is_ollama_available

        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch("aer.utils.ollama_client.requests.get", return_value=mock_response):
            result = is_ollama_available()
        assert isinstance(result, bool)


class TestOllamaCallUnavailable:
    def test_raises_runtime_error_when_unavailable(self) -> None:
        from aer.utils.ollama_client import ollama_call

        with patch("aer.utils.ollama_client.is_ollama_available", return_value=False):
            with pytest.raises(RuntimeError, match="Ollama unavailable"):
                ollama_call("test prompt")

    def test_caller_can_catch_and_fall_back(self) -> None:
        """Pattern test: callers must wrap ollama_call in try/except RuntimeError."""
        from aer.utils.ollama_client import OLLAMA_DEFAULT_MODEL, ollama_call

        fallback_triggered = False

        def caller_with_fallback(prompt: str) -> str:
            nonlocal fallback_triggered
            try:
                return ollama_call(prompt, model=OLLAMA_DEFAULT_MODEL)
            except RuntimeError:
                fallback_triggered = True
                return "haiku_fallback_result"

        with patch("aer.utils.ollama_client.is_ollama_available", return_value=False):
            result = caller_with_fallback("analyze this")

        assert fallback_triggered is True
        assert result == "haiku_fallback_result"


class TestOllamaCallOpenAIPath:
    """Tests for non-json_schema path (uses OpenAI-compatible client)."""

    def _mock_openai_response(self, content: str) -> MagicMock:
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_returns_content_string(self) -> None:
        from aer.utils.ollama_client import ollama_call

        mock_resp = self._mock_openai_response("analysis result")
        with (
            patch("aer.utils.ollama_client.is_ollama_available", return_value=True),
            patch("aer.utils.ollama_client.OpenAI") as mock_openai_cls,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_openai_cls.return_value = mock_client

            result = ollama_call("analyze AAPL")

        assert result == "analysis result"

    def test_sets_temperature_zero(self) -> None:
        from aer.utils.ollama_client import ollama_call

        mock_resp = self._mock_openai_response("ok")
        with (
            patch("aer.utils.ollama_client.is_ollama_available", return_value=True),
            patch("aer.utils.ollama_client.OpenAI") as mock_openai_cls,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_openai_cls.return_value = mock_client

            ollama_call("test", max_tokens=100)
            call_kwargs = mock_client.chat.completions.create.call_args

        assert (
            call_kwargs.kwargs.get("temperature") == 0
            or (call_kwargs.args and call_kwargs.args[0].get("temperature") == 0)
            or call_kwargs[1].get("temperature") == 0
        )

    def test_max_tokens_forwarded(self) -> None:
        from aer.utils.ollama_client import ollama_call

        mock_resp = self._mock_openai_response("ok")
        with (
            patch("aer.utils.ollama_client.is_ollama_available", return_value=True),
            patch("aer.utils.ollama_client.OpenAI") as mock_openai_cls,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_resp
            mock_openai_cls.return_value = mock_client

            ollama_call("test", max_tokens=42)
            call_kwargs = mock_client.chat.completions.create.call_args

        assert (
            call_kwargs.kwargs.get("max_tokens") == 42
            or (call_kwargs.args and call_kwargs.args[0].get("max_tokens") == 42)
            or call_kwargs[1].get("max_tokens") == 42
        )

    def test_system_prompt_included_when_provided(self) -> None:
        from aer.utils.ollama_client import ollama_call

        mock_resp = self._mock_openai_response("ok")
        captured: list = []

        def capture_call(**kwargs):
            captured.append(kwargs.get("messages", []))
            return mock_resp

        with (
            patch("aer.utils.ollama_client.is_ollama_available", return_value=True),
            patch("aer.utils.ollama_client.OpenAI") as mock_openai_cls,
        ):
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = capture_call
            mock_openai_cls.return_value = mock_client

            ollama_call("user msg", system="You are an analyst.")

        assert len(captured) == 1
        roles = [m["role"] for m in captured[0]]
        assert "system" in roles


class TestOllamaCallJsonSchemaPath:
    """Tests for json_schema path (uses native ollama library)."""

    def test_json_schema_returns_valid_json_string(self) -> None:
        from aer.utils.ollama_client import ollama_call

        valid_json = json.dumps({"signal": "BULLISH", "confidence": 0.8})
        mock_msg = MagicMock()
        mock_msg.content = valid_json
        mock_response = MagicMock()
        mock_response.message = mock_msg

        schema = {"type": "object", "properties": {"signal": {"type": "string"}}}

        with (
            patch("aer.utils.ollama_client.is_ollama_available", return_value=True),
            patch.dict("sys.modules", {"ollama": MagicMock()}),
        ):
            import sys

            sys.modules["ollama"].chat.return_value = mock_response

            result = ollama_call("classify signal", json_schema=schema)

        parsed = json.loads(result)
        assert parsed["signal"] == "BULLISH"

    def test_json_schema_path_uses_native_ollama_library(self) -> None:
        from aer.utils.ollama_client import ollama_call

        mock_msg = MagicMock()
        mock_msg.content = '{"label": "BUY"}'
        mock_response = MagicMock()
        mock_response.message = mock_msg

        schema = {"type": "object"}

        with (
            patch("aer.utils.ollama_client.is_ollama_available", return_value=True),
            patch.dict("sys.modules", {"ollama": MagicMock()}),
        ):
            import sys

            mock_ollama = sys.modules["ollama"]
            mock_ollama.chat.return_value = mock_response

            ollama_call("test", json_schema=schema)

            assert mock_ollama.chat.called
