"""Tests for LLM cost control, feature gates, and determinism enforcement.

Verification: temperature=0 enforcement, json_mode enforcement,
HTTP 529 handling, and feature flag gates for instagram/layoffs agents.
"""

from unittest.mock import MagicMock, patch

import pytest

from aer.config import Settings
from aer.utils.cost_thresholds import (
    MAX_LLM_CALLS_PER_AGENT_RUN,
    MAX_TWITTER_COST_PER_DAY_USD,
)
from aer.utils.llm_client import complete


class TestCostConstants:
    """Verify all cost constants are defined correctly."""

    def test_max_twitter_cost_per_day_defined(self) -> None:
        """MAX_TWITTER_COST_PER_DAY_USD exists and has correct value."""
        assert MAX_TWITTER_COST_PER_DAY_USD == 5.0
        assert isinstance(MAX_TWITTER_COST_PER_DAY_USD, float)

    def test_max_llm_calls_per_agent_run_defined(self) -> None:
        """MAX_LLM_CALLS_PER_AGENT_RUN exists and has correct value."""
        assert MAX_LLM_CALLS_PER_AGENT_RUN == 25
        assert isinstance(MAX_LLM_CALLS_PER_AGENT_RUN, int)

    def test_all_constants_are_positive(self) -> None:
        """All cost constants must be positive (no zero or negative values)."""
        assert MAX_TWITTER_COST_PER_DAY_USD > 0.0
        assert MAX_LLM_CALLS_PER_AGENT_RUN > 0


class TestFeatureFlags:
    """Verify feature flags default to False."""

    def test_agent_enabled_instagram_defaults_false(self) -> None:
        """AGENT_ENABLED_INSTAGRAM defaults to False."""
        settings = Settings()
        assert hasattr(settings, "agent_enabled_instagram")
        assert settings.agent_enabled_instagram is False

    def test_agent_enabled_layoffs_defaults_false(self) -> None:
        """AGENT_ENABLED_LAYOFFS defaults to False."""
        settings = Settings()
        assert hasattr(settings, "agent_enabled_layoffs")
        assert settings.agent_enabled_layoffs is False


class TestExtractionTierTemperature:
    """Verify EXTRACTION tier enforces temperature=0."""

    @patch("aer.utils.llm_client._attempt_provider")
    @patch(
        "aer.utils.llm_client.TIER_CONFIG",
        {
            "EXTRACTION": MagicMock(
                primary_provider="openrouter",
                primary_model="openrouter/test",
                secondary_provider="openrouter",
                secondary_model="test",
            ),
            "ANALYSIS": MagicMock(),
            "SYNTHESIS": MagicMock(),
        },
    )
    @patch("aer.utils.llm_client._estimate_tokens", return_value=100)
    def test_extraction_tier_forces_temperature_zero(
        self,
        mock_estimate: MagicMock,
        mock_provider: MagicMock,
    ) -> None:
        """EXTRACTION tier overrides any provided temperature to 0.0."""
        mock_provider.return_value = '{"result": "extracted"}'

        response = complete(
            system_prompt="Extract data.",
            user_message="Data: test",
            tier="EXTRACTION",
            max_tokens=100,
            temperature=1.0,  # User provides non-zero, should be overridden
        )

        # Verify the call was made with temperature=0.0
        assert mock_provider.called
        call_kwargs = mock_provider.call_args[1]
        assert call_kwargs["temperature"] == 0.0
        assert response.content == '{"result": "extracted"}'

    @patch("aer.utils.llm_client._attempt_provider")
    @patch(
        "aer.utils.llm_client.TIER_CONFIG",
        {
            "EXTRACTION": MagicMock(
                primary_provider="openrouter",
                primary_model="test",
                secondary_provider="openrouter",
                secondary_model="test",
            ),
            "ANALYSIS": MagicMock(),
            "SYNTHESIS": MagicMock(),
        },
    )
    @patch("aer.utils.llm_client._estimate_tokens", return_value=100)
    def test_extraction_tier_respects_zero_temperature(
        self,
        mock_estimate: MagicMock,
        mock_provider: MagicMock,
    ) -> None:
        """EXTRACTION tier respects if user explicitly provides temperature=0.0."""
        mock_provider.return_value = '{"key": "value"}'

        complete(
            system_prompt="Extract.",
            user_message="Data: test",
            tier="EXTRACTION",
            max_tokens=100,
            temperature=0.0,  # Already zero, should remain zero
        )

        assert mock_provider.called
        call_kwargs = mock_provider.call_args[1]
        assert call_kwargs["temperature"] == 0.0


class TestExtractionTierJsonMode:
    """Verify EXTRACTION tier enforces json_mode=True."""

    @patch("aer.utils.llm_client._attempt_provider")
    @patch(
        "aer.utils.llm_client.TIER_CONFIG",
        {
            "EXTRACTION": MagicMock(
                primary_provider="openrouter",
                primary_model="test",
                secondary_provider="openrouter",
                secondary_model="test",
            ),
            "ANALYSIS": MagicMock(),
            "SYNTHESIS": MagicMock(),
        },
    )
    @patch("aer.utils.llm_client._estimate_tokens", return_value=100)
    def test_extraction_tier_forces_json_mode(
        self,
        mock_estimate: MagicMock,
        mock_provider: MagicMock,
    ) -> None:
        """EXTRACTION tier enforces json_mode=True regardless of caller setting."""
        mock_provider.return_value = '{"extracted": true}'

        # User passes json_mode=False (incorrect), should be forced to True
        response = complete(
            system_prompt="Extract JSON.",
            user_message="Data: test",
            tier="EXTRACTION",
            max_tokens=100,
            json_mode=False,  # Should be overridden to True
        )

        assert mock_provider.called
        call_kwargs = mock_provider.call_args[1]
        assert call_kwargs["json_mode"] is True
        assert response.content == '{"extracted": true}'


class TestHttp529Handling:
    """Verify HTTP 529 is treated as retriable like 429."""

    @patch("aer.utils.llm_client.time.sleep")
    @patch("aer.utils.llm_client._call_free_provider")
    def test_http_529_triggers_retry(self, mock_call: MagicMock, mock_sleep: MagicMock) -> None:
        """HTTP 529 error is detected and triggers retry logic."""
        from aer.utils.llm_client import _attempt_provider

        # Simulate: first call raises 529 error, second call succeeds
        mock_call.side_effect = [
            Exception("HTTP error 529: Overloaded"),
            '{"success": true}',  # Second attempt succeeds
        ]
        failure_chain: list[str] = []

        result = _attempt_provider(
            provider="openrouter",
            model="test-model",
            system_prompt="Test system.",
            user_message="Test user.",
            max_tokens=100,
            json_mode=False,
            failure_chain=failure_chain,
            temperature=0.0,
        )

        # Verify retry happened: two calls made
        assert mock_call.call_count == 2
        # Verify result is from second (successful) call
        assert result == '{"success": true}'
        # Verify failure logged from first attempt
        assert any("529" in str(f) or "Overloaded" in str(f) for f in failure_chain)

    @patch("aer.utils.llm_client.time.sleep")
    @patch("aer.utils.llm_client._call_free_provider")
    def test_overloaded_keyword_triggers_retry(self, mock_call: MagicMock, mock_sleep: MagicMock) -> None:
        """'Overloaded' keyword in error message is detected and triggers retry."""
        from aer.utils.llm_client import _attempt_provider

        mock_call.side_effect = [
            Exception("Anthropic API is overloaded"),
            '{"data": "extracted"}',
        ]
        failure_chain: list[str] = []

        result = _attempt_provider(
            provider="anthropic",
            model="fallback",
            system_prompt="Test.",
            user_message="Data.",
            max_tokens=100,
            json_mode=False,
            failure_chain=failure_chain,
            temperature=0.0,
        )

        # Verify retry occurred
        assert mock_call.call_count == 2
        assert result == '{"data": "extracted"}'


class TestAnalysisTierTemperature:
    """Verify ANALYSIS tier allows custom temperature (not forced to 0)."""

    @patch("aer.utils.llm_client._attempt_provider")
    @patch(
        "aer.utils.llm_client.TIER_CONFIG",
        {
            "EXTRACTION": MagicMock(),
            "ANALYSIS": MagicMock(
                primary_provider="openrouter",
                primary_model="test",
                secondary_provider="openrouter",
                secondary_model="test",
            ),
            "SYNTHESIS": MagicMock(),
        },
    )
    @patch("aer.utils.llm_client._estimate_tokens", return_value=100)
    def test_analysis_tier_preserves_temperature(
        self,
        mock_estimate: MagicMock,
        mock_provider: MagicMock,
    ) -> None:
        """ANALYSIS tier does NOT force temperature=0; respects caller's value."""
        mock_provider.return_value = "Analysis result"

        response = complete(
            system_prompt="Analyze.",
            user_message="Data.",
            tier="ANALYSIS",
            max_tokens=500,
            temperature=0.7,  # Should NOT be overridden for ANALYSIS
        )

        assert mock_provider.called
        call_kwargs = mock_provider.call_args[1]
        assert call_kwargs["temperature"] == 0.7  # Preserved as-is
        assert response.content == "Analysis result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
