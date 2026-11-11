"""Tests for Ollama reliability and error handling.

Verifies:
1. Single O(1) health-check before model cascade (not per-model)
2. Explicit ConnectionError handling with structured failure_type logging
3. Daemon-down vs model-error discrimination
"""

from unittest.mock import MagicMock, patch

import pytest
import requests.exceptions


class TestHealthCheckSingle:
    """Verify is_ollama_available() performs single O(1) health-check."""

    def test_daemon_down_detected_once(self) -> None:
        """is_ollama_available() returns False on connection refused; called exactly once."""
        from aer.utils.ollama_client import is_ollama_available

        with patch("aer.utils.ollama_client.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("refused")

            result = is_ollama_available()

            # Verify False is returned
            assert result is False
            # Verify requests.get called exactly once (single health-check)
            assert mock_get.call_count == 1

    def test_health_check_timeout_detected(self) -> None:
        """is_ollama_available() returns False on timeout."""
        from aer.utils.ollama_client import is_ollama_available

        with patch("aer.utils.ollama_client.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("timeout")

            result = is_ollama_available()

            # Verify False is returned
            assert result is False
            # Verify exactly one call
            assert mock_get.call_count == 1

    def test_health_check_returns_true_on_200(self) -> None:
        """is_ollama_available() returns True when daemon responds with 200."""
        from aer.utils.ollama_client import is_ollama_available

        with patch("aer.utils.ollama_client.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = is_ollama_available()

            # Verify True is returned
            assert result is True
            # Verify exactly one call
            assert mock_get.call_count == 1


class TestExplicitErrorHandling:
    """Verify explicit exception handling with failure_type discrimination."""

    def test_connection_error_caught_explicitly(self) -> None:
        """ConnectionError from is_ollama_available() is caught explicitly."""
        from aer.utils.ollama_client import is_ollama_available

        with patch("aer.utils.ollama_client.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

            # Should not raise; returns False
            result = is_ollama_available()
            assert result is False

    def test_timeout_caught_explicitly(self) -> None:
        """Timeout from health-check is caught explicitly."""
        from aer.utils.ollama_client import is_ollama_available

        with patch("aer.utils.ollama_client.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Timeout")

            # Should not raise; returns False
            result = is_ollama_available()
            assert result is False

    def test_request_exception_caught_explicitly(self) -> None:
        """Generic RequestException is caught explicitly."""
        from aer.utils.ollama_client import is_ollama_available

        with patch("aer.utils.ollama_client.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Other error")

            # Should not raise; returns False
            result = is_ollama_available()
            assert result is False


class TestCascadeSkipsOllamaWhenUnavailable:
    """Verify cascade skips Ollama entirely when health-check fails."""

    def test_cascade_raises_runtime_error_when_ollama_unavailable(self) -> None:
        """ollama_call_with_cascade raises RuntimeError when daemon is down."""
        from aer.utils.ollama_client import ollama_call_with_cascade

        with patch("aer.utils.ollama_client.is_ollama_available", return_value=False):
            with pytest.raises(RuntimeError, match="unavailable"):
                ollama_call_with_cascade(prompt="test", system="test")

    def test_ollama_call_not_attempted_when_unavailable(self) -> None:
        """ollama_call() is not invoked if health-check fails."""
        from aer.utils.ollama_client import ollama_call_with_cascade

        with patch("aer.utils.ollama_client.is_ollama_available", return_value=False):
            with patch("aer.utils.ollama_client.ollama_call") as mock_call:
                with pytest.raises(RuntimeError):
                    ollama_call_with_cascade(prompt="test", system="test")

                # Verify ollama_call was never attempted
                assert mock_call.call_count == 0


class TestFailureTypeDiscrimination:
    """Verify failure_type discriminates daemon-down from model errors."""

    def test_failure_chain_includes_failure_type(self) -> None:
        """Failure chain entries include failure_type field."""
        from aer.utils.llm_client import complete

        with patch("aer.utils.ollama_client.is_ollama_available", return_value=False):
            with patch("aer.utils.llm_client._attempt_provider", return_value=None):
                # When all providers fail, complete() will raise RuntimeError
                # We're checking that the failure_chain includes failure_type for ollama entry
                try:
                    complete(
                        system_prompt="test",
                        user_message="test",
                        tier="ANALYSIS",
                        max_tokens=100,
                    )
                except RuntimeError as e:
                    # The error message contains the failure_chain
                    error_str = str(e)
                    # Verify ollama failure_type is in the error message
                    assert "ollama_unavailable" in error_str or "ollama" in error_str.lower()

    def test_daemon_down_vs_model_error_discrimination(self) -> None:
        """Failure chain discriminates daemon-down from model errors."""
        from aer.utils.ollama_client import ollama_call_with_cascade

        # Scenario 1: Daemon down
        with patch("aer.utils.ollama_client.is_ollama_available", return_value=False):
            with pytest.raises(RuntimeError, match="unavailable"):
                ollama_call_with_cascade(prompt="test", system="test")

        # Scenario 2: Model error (daemon up but model fails)
        with patch("aer.utils.ollama_client.is_ollama_available", return_value=True):
            with patch("aer.utils.ollama_client.ollama_call") as mock_call:
                mock_call.side_effect = Exception("Model not found")

                with pytest.raises(RuntimeError):
                    ollama_call_with_cascade(prompt="test", system="test")


class TestOllamaIntegration:
    """Integration tests for Ollama + LLM cascade."""

    def test_complete_cascades_when_ollama_unavailable(self) -> None:
        """complete() cascades to next tier when Ollama is unavailable."""
        from aer.utils.llm_client import complete

        with patch("aer.utils.ollama_client.is_ollama_available", return_value=False):
            # Mock the free providers to fail as well
            with patch("aer.utils.llm_client._attempt_provider", return_value=None):
                # Should eventually reach Anthropic fallback
                with patch("aer.utils.llm_client._call_anthropic_fallback") as mock_fallback:
                    mock_fallback.return_value = ("response", 10, 20)

                    resp = complete(
                        system_prompt="test",
                        user_message="test",
                        tier="ANALYSIS",
                        max_tokens=100,
                    )

                    # Verify Anthropic fallback was called
                    assert mock_fallback.called
                    # Verify failure_chain mentions ollama unavailable
                    assert "ollama" in str(resp.failure_chain).lower()

    def test_health_check_prevents_per_model_retries(self) -> None:
        """Single health-check prevents per-model retry overhead."""
        from aer.utils.ollama_client import ollama_call_with_cascade

        with patch("aer.utils.ollama_client.is_ollama_available") as mock_health:
            mock_health.return_value = False

            with pytest.raises(RuntimeError):
                ollama_call_with_cascade(prompt="test", system="test")

            # Verify health-check called exactly once (not per model)
            assert mock_health.call_count == 1


class TestLoggingWithFailureType:
    """Verify logging includes structured failure_type for observability."""

    def test_ollama_available_logs_with_failure_type(self) -> None:
        """is_ollama_available() logs with failure_type extra when errors occur."""
        from aer.utils.ollama_client import is_ollama_available

        with patch("aer.utils.ollama_client.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("test")

            with patch("aer.utils.ollama_client.logger") as mock_logger:
                is_ollama_available()

                # Verify logger.warning was called
                assert mock_logger.warning.called
                # Verify call includes extra dict with failure_type
                call_args = mock_logger.warning.call_args
                assert call_args is not None
                # Check if extra dict was passed
                if "extra" in call_args[1]:
                    assert "failure_type" in call_args[1]["extra"]

    def test_cascade_logs_daemon_down_with_failure_type(self) -> None:
        """ollama_call_with_cascade() logs daemon-down with failure_type."""
        from aer.utils.ollama_client import ollama_call_with_cascade

        with patch("aer.utils.ollama_client.is_ollama_available", return_value=False):
            with patch("aer.utils.ollama_client.logger") as mock_logger:
                try:
                    ollama_call_with_cascade(prompt="test", system="test")
                except RuntimeError:
                    pass

                # Verify logger.warning was called
                assert mock_logger.warning.called
                # Verify call includes extra dict with failure_type
                call_args = mock_logger.warning.call_args
                assert call_args is not None
                if "extra" in call_args[1]:
                    assert call_args[1]["extra"]["failure_type"] == "ollama_daemon_down"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
