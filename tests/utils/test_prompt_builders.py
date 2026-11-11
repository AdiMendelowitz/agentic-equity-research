"""Tests for src/utils/prompt_builders.py and src/utils/python_replacements.py."""

from unittest.mock import MagicMock


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        from aer.utils.prompt_builders import estimate_tokens

        assert estimate_tokens("") == 0

    def test_four_chars_is_one_token(self) -> None:
        from aer.utils.prompt_builders import estimate_tokens

        assert estimate_tokens("abcd") == 1

    def test_twelve_chars_is_three_tokens(self) -> None:
        from aer.utils.prompt_builders import estimate_tokens

        assert estimate_tokens("abcdefghijkl") == 3

    def test_non_ascii_chars(self) -> None:
        from aer.utils.prompt_builders import estimate_tokens

        # function counts raw chars, not bytes
        result = estimate_tokens("αβγδ")
        assert result == 1


class TestTruncateToTokens:
    def test_no_truncation_needed(self) -> None:
        from aer.utils.prompt_builders import truncate_to_tokens

        assert truncate_to_tokens("hello", 10) == "hello"

    def test_truncates_to_exact_length(self) -> None:
        from aer.utils.prompt_builders import truncate_to_tokens

        result = truncate_to_tokens("abcdefghij", max_tokens=2)
        assert result == "abcdefgh [truncated]"  # 2 tokens * 4 chars = 8 chars + notice

    def test_zero_tokens_returns_empty(self) -> None:
        from aer.utils.prompt_builders import truncate_to_tokens

        assert truncate_to_tokens("hello world", 0) == ""


class TestExtractFields:
    def test_from_dict(self) -> None:
        from aer.utils.prompt_builders import extract_fields

        obj = {"a": 1, "b": 2, "c": None}
        result = extract_fields(obj, ["a", "b", "c"])
        assert result == {"a": 1, "b": 2}

    def test_from_object(self) -> None:
        from aer.utils.prompt_builders import extract_fields

        obj = MagicMock()
        obj.x = 10
        obj.y = None
        result = extract_fields(obj, ["x", "y"])
        assert result == {"x": 10}

    def test_missing_dict_key_skipped(self) -> None:
        from aer.utils.prompt_builders import extract_fields

        result = extract_fields({"a": 1}, ["a", "missing"])
        assert result == {"a": 1}

    def test_empty_fields_list(self) -> None:
        from aer.utils.prompt_builders import extract_fields

        assert extract_fields({"a": 1}, []) == {}


class TestBuildSynthesisUserMessage:
    def _make_output(self, name: str, content: str, confidence: float) -> MagicMock:
        m = MagicMock()
        m.content = content
        m.confidence = confidence
        return m

    def test_ticker_in_message(self) -> None:
        from aer.utils.prompt_builders import build_synthesis_user_message

        msg = build_synthesis_user_message("AAPL", {})
        assert "AAPL" in msg

    def test_agent_section_header_present(self) -> None:
        from aer.utils.prompt_builders import build_synthesis_user_message

        outputs = {"fundamental": self._make_output("fundamental", "strong fundamentals", 0.8)}
        msg = build_synthesis_user_message("AAPL", outputs)
        assert "=== FUNDAMENTAL" in msg

    def test_confidence_formatted_as_percent(self) -> None:
        from aer.utils.prompt_builders import build_synthesis_user_message

        outputs = {"macro": self._make_output("macro", "macro analysis", 0.75)}
        msg = build_synthesis_user_message("AAPL", outputs)
        assert "75%" in msg

    def test_content_truncated_to_max_chars(self) -> None:
        from aer.utils.prompt_builders import build_synthesis_user_message

        long_content = "x" * 5000
        outputs = {"fundamental": self._make_output("fundamental", long_content, 0.8)}
        msg = build_synthesis_user_message("AAPL", outputs, max_chars_per_agent=2000)
        assert "x" * 2000 in msg
        assert "x" * 2001 not in msg

    def test_none_agent_skipped(self) -> None:
        from aer.utils.prompt_builders import build_synthesis_user_message

        outputs = {"fundamental": None, "macro": self._make_output("macro", "macro text", 0.7)}
        msg = build_synthesis_user_message("AAPL", outputs)
        assert "FUNDAMENTAL" not in msg
        assert "MACRO" in msg

    def test_closing_instruction_present(self) -> None:
        from aer.utils.prompt_builders import build_synthesis_user_message

        msg = build_synthesis_user_message("AAPL", {})
        assert "structured investment recommendation" in msg

    def test_default_max_chars_is_2000(self) -> None:
        from aer.utils.prompt_builders import build_synthesis_user_message

        long_content = "y" * 3000
        outputs = {"technical": self._make_output("technical", long_content, 0.6)}
        msg = build_synthesis_user_message("AAPL", outputs)
        assert "y" * 2000 in msg
        assert "y" * 2001 not in msg


class TestPythonReplacements:
    def test_signal_label_strong_buy(self) -> None:
        from aer.utils.python_replacements import signal_label

        assert signal_label(6.0) == "STRONG_BUY"
        assert signal_label(10.0) == "STRONG_BUY"

    def test_signal_label_buy(self) -> None:
        from aer.utils.python_replacements import signal_label

        assert signal_label(3.0) == "BUY"
        assert signal_label(5.9) == "BUY"

    def test_signal_label_watch(self) -> None:
        from aer.utils.python_replacements import signal_label

        assert signal_label(1.5) == "WATCH"
        assert signal_label(2.9) == "WATCH"

    def test_signal_label_noise(self) -> None:
        from aer.utils.python_replacements import signal_label

        assert signal_label(0.0) == "NOISE"
        assert signal_label(1.4) == "NOISE"

    def test_classify_vix_low_fear(self) -> None:
        from aer.utils.python_replacements import classify_vix_regime

        assert classify_vix_regime(10.0) == "LOW_FEAR"
        assert classify_vix_regime(14.9) == "LOW_FEAR"

    def test_classify_vix_normal(self) -> None:
        from aer.utils.python_replacements import classify_vix_regime

        assert classify_vix_regime(15.0) == "NORMAL"
        assert classify_vix_regime(19.9) == "NORMAL"

    def test_classify_vix_elevated(self) -> None:
        from aer.utils.python_replacements import classify_vix_regime

        assert classify_vix_regime(20.0) == "ELEVATED"
        assert classify_vix_regime(29.9) == "ELEVATED"

    def test_classify_vix_high_fear(self) -> None:
        from aer.utils.python_replacements import classify_vix_regime

        assert classify_vix_regime(30.0) == "HIGH_FEAR"
        assert classify_vix_regime(80.0) == "HIGH_FEAR"

    def test_pc_ratio_lean_bullish(self) -> None:
        from aer.utils.python_replacements import pc_ratio_lean

        assert pc_ratio_lean(0.8) == "bullish"
        assert pc_ratio_lean(1.0) == "bullish"

    def test_pc_ratio_lean_bearish(self) -> None:
        from aer.utils.python_replacements import pc_ratio_lean

        assert pc_ratio_lean(1.1) == "bearish"
        assert pc_ratio_lean(2.5) == "bearish"


class TestBuildSignalPrompt:
    def test_excludes_none_indicators(self) -> None:
        from aer.utils.prompt_builders import build_signal_prompt

        _, user = build_signal_prompt(
            "AAPL",
            signals={"trend": "BULLISH"},
            indicators={"RSI": 55.1, "MACD": None, "ADX": 28.3},
        )
        assert "MACD" not in user
        assert "RSI" in user
        assert "ADX" in user

    def test_appends_json_only_suffix(self) -> None:
        from aer.utils.prompt_builders import build_signal_prompt

        _, user = build_signal_prompt("TSLA", signals={}, indicators={})
        assert "Return JSON only" in user
        assert "No prose" in user
        assert "No markdown fences" in user

    def test_ticker_in_user_message(self) -> None:
        from aer.utils.prompt_builders import build_signal_prompt

        _, user = build_signal_prompt("NVDA", signals={}, indicators={})
        assert "NVDA" in user

    def test_returns_tuple_of_two_strings(self) -> None:
        from aer.utils.prompt_builders import build_signal_prompt

        result = build_signal_prompt("MSFT", signals={}, indicators={})
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(s, str) for s in result)
