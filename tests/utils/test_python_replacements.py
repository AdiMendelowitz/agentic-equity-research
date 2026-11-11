"""Tests for src/utils/python_replacements.py.

Regression guard: every function here must remain pure Python with no LLM calls.
Prompt-building helpers (estimate_tokens, truncate_to_tokens, extract_fields) live
in prompt_builders.py — those tests are in test_prompt_builders.py.
"""

# ---------------------------------------------------------------------------
# classify_vix_regime
# ---------------------------------------------------------------------------


class TestClassifyVixRegime:
    def test_low_fear_below_15(self) -> None:
        from aer.utils.python_replacements import classify_vix_regime

        assert classify_vix_regime(10.0) == "LOW_FEAR"
        assert classify_vix_regime(14.99) == "LOW_FEAR"

    def test_normal_between_15_and_20(self) -> None:
        from aer.utils.python_replacements import classify_vix_regime

        assert classify_vix_regime(15.0) == "NORMAL"
        assert classify_vix_regime(19.99) == "NORMAL"

    def test_elevated_between_20_and_30(self) -> None:
        from aer.utils.python_replacements import classify_vix_regime

        assert classify_vix_regime(20.0) == "ELEVATED"
        assert classify_vix_regime(29.99) == "ELEVATED"

    def test_high_fear_at_30_and_above(self) -> None:
        from aer.utils.python_replacements import classify_vix_regime

        assert classify_vix_regime(30.0) == "HIGH_FEAR"
        assert classify_vix_regime(80.0) == "HIGH_FEAR"

    def test_all_boundaries_are_distinct(self) -> None:
        from aer.utils.python_replacements import classify_vix_regime

        labels = {classify_vix_regime(v) for v in (10.0, 17.0, 25.0, 45.0)}
        assert labels == {"LOW_FEAR", "NORMAL", "ELEVATED", "HIGH_FEAR"}


# ---------------------------------------------------------------------------
# pc_ratio_lean
# ---------------------------------------------------------------------------


class TestPcRatioLean:
    def test_ratio_above_1_is_bearish(self) -> None:
        from aer.utils.python_replacements import pc_ratio_lean

        assert pc_ratio_lean(1.1) == "bearish"
        assert pc_ratio_lean(2.0) == "bearish"

    def test_ratio_at_or_below_1_is_bullish(self) -> None:
        from aer.utils.python_replacements import pc_ratio_lean

        assert pc_ratio_lean(1.0) == "bullish"
        assert pc_ratio_lean(0.5) == "bullish"
        assert pc_ratio_lean(0.0) == "bullish"


# ---------------------------------------------------------------------------
# signal_label
# ---------------------------------------------------------------------------


class TestSignalLabel:
    def test_strong_buy_at_and_above_threshold(self) -> None:
        from aer.utils.python_replacements import signal_label

        assert signal_label(6.0) == "STRONG_BUY"
        assert signal_label(10.0) == "STRONG_BUY"

    def test_buy_between_3_and_6(self) -> None:
        from aer.utils.python_replacements import signal_label

        assert signal_label(3.0) == "BUY"
        assert signal_label(5.9) == "BUY"

    def test_watch_between_1_5_and_3(self) -> None:
        from aer.utils.python_replacements import signal_label

        assert signal_label(1.5) == "WATCH"
        assert signal_label(2.9) == "WATCH"

    def test_noise_below_threshold(self) -> None:
        from aer.utils.python_replacements import signal_label

        assert signal_label(0.0) == "NOISE"
        assert signal_label(1.4) == "NOISE"
        assert signal_label(-1.0) == "NOISE"


# ---------------------------------------------------------------------------
# signal_action_from_classification
# ---------------------------------------------------------------------------


class TestSignalActionFromClassification:
    def test_bullish_maps_to_buy(self) -> None:
        from aer.utils.python_replacements import signal_action_from_classification

        assert signal_action_from_classification("BULLISH") == "BUY"
        assert signal_action_from_classification("bullish") == "BUY"

    def test_bearish_maps_to_sell(self) -> None:
        from aer.utils.python_replacements import signal_action_from_classification

        assert signal_action_from_classification("BEARISH") == "SELL"
        assert signal_action_from_classification("bearish") == "SELL"

    def test_neutral_and_unknown_map_to_hold(self) -> None:
        from aer.utils.python_replacements import signal_action_from_classification

        assert signal_action_from_classification("NEUTRAL") == "HOLD"
        assert signal_action_from_classification("unknown") == "HOLD"
        assert signal_action_from_classification("") == "HOLD"


# ---------------------------------------------------------------------------
# derive_macro_outlook
# ---------------------------------------------------------------------------


class TestDeriveMacroOutlook:
    def test_bullish_bias_is_favorable(self) -> None:
        from aer.utils.python_replacements import derive_macro_outlook

        assert derive_macro_outlook({"equity_bias": "BULLISH"}) == "FAVORABLE"
        assert derive_macro_outlook({"equity_bias": "STRONGLY BULLISH"}) == "FAVORABLE"

    def test_bearish_bias_is_unfavorable(self) -> None:
        from aer.utils.python_replacements import derive_macro_outlook

        assert derive_macro_outlook({"equity_bias": "BEARISH"}) == "UNFAVORABLE"
        assert derive_macro_outlook({"equity_bias": "MILDLY BEARISH"}) == "UNFAVORABLE"

    def test_neutral_and_unknown_are_neutral(self) -> None:
        from aer.utils.python_replacements import derive_macro_outlook

        assert derive_macro_outlook({"equity_bias": "NEUTRAL"}) == "NEUTRAL"
        assert derive_macro_outlook({}) == "NEUTRAL"
        assert derive_macro_outlook({"equity_bias": ""}) == "NEUTRAL"


# ---------------------------------------------------------------------------
# derive_sentiment_consensus
# ---------------------------------------------------------------------------


class TestDeriveSentimentConsensus:
    def test_two_bullish_sources_return_bullish(self) -> None:
        from aer.utils.python_replacements import derive_sentiment_consensus

        sources = {"aw": "BULLISH", "st": "BULLISH", "news": "NEUTRAL"}
        assert derive_sentiment_consensus(sources) == "BULLISH"

    def test_two_bearish_sources_return_bearish(self) -> None:
        from aer.utils.python_replacements import derive_sentiment_consensus

        sources = {"aw": "BEARISH", "st": "BEARISH", "news": "BULLISH"}
        assert derive_sentiment_consensus(sources) == "BEARISH"

    def test_two_neutral_sources_return_neutral(self) -> None:
        from aer.utils.python_replacements import derive_sentiment_consensus

        sources = {"aw": "NEUTRAL", "st": "NEUTRAL"}
        assert derive_sentiment_consensus(sources) == "NEUTRAL"

    def test_all_different_returns_mixed(self) -> None:
        from aer.utils.python_replacements import derive_sentiment_consensus

        sources = {"aw": "BULLISH", "st": "BEARISH", "news": "NEUTRAL"}
        assert derive_sentiment_consensus(sources) == "MIXED"

    def test_single_source_returns_mixed(self) -> None:
        from aer.utils.python_replacements import derive_sentiment_consensus

        assert derive_sentiment_consensus({"aw": "BULLISH"}) == "MIXED"


# ---------------------------------------------------------------------------
# derive_positioning_signal
# ---------------------------------------------------------------------------


class TestDerivePositioningSignal:
    def test_high_put_call_and_high_beat_rate_is_bullish(self) -> None:
        from aer.utils.python_replacements import derive_positioning_signal

        assert derive_positioning_signal(1.3, 0.1, 0.8) == "BULLISH"

    def test_low_put_call_and_low_beat_rate_is_bearish(self) -> None:
        from aer.utils.python_replacements import derive_positioning_signal

        assert derive_positioning_signal(0.4, 0.5, 0.3) == "BEARISH"

    def test_mixed_signals_return_neutral(self) -> None:
        from aer.utils.python_replacements import derive_positioning_signal

        assert derive_positioning_signal(1.0, 0.1, 0.55) == "NEUTRAL"

    def test_none_inputs_return_neutral(self) -> None:
        from aer.utils.python_replacements import derive_positioning_signal

        assert derive_positioning_signal(None, None, None) == "NEUTRAL"

    def test_none_beat_rate_defaults_to_neutral(self) -> None:
        from aer.utils.python_replacements import derive_positioning_signal

        # pc > 1.2 but beat defaults to 0.5, not > 0.7 → NEUTRAL
        assert derive_positioning_signal(1.5, None, None) == "NEUTRAL"


# ---------------------------------------------------------------------------
# compute_trend
# ---------------------------------------------------------------------------


class TestComputeTrend:
    def test_rising(self) -> None:
        import pandas as pd

        from aer.utils.python_replacements import compute_trend

        # delta = 100 - 10 = 90; mean ≈ 55; threshold = 0.55 → well above
        series = pd.Series([10.0, 50.0, 100.0])
        assert compute_trend(series) == "RISING"

    def test_falling(self) -> None:
        import pandas as pd

        from aer.utils.python_replacements import compute_trend

        series = pd.Series([100.0, 50.0, 10.0])
        assert compute_trend(series) == "FALLING"

    def test_flat(self) -> None:
        import pandas as pd

        from aer.utils.python_replacements import compute_trend

        # delta = 0; below any positive threshold
        series = pd.Series([50.0, 50.0, 50.0])
        assert compute_trend(series) == "FLAT"

    def test_insufficient_data_single_non_null(self) -> None:
        import pandas as pd

        from aer.utils.python_replacements import compute_trend

        series = pd.Series([float("nan"), 5.0])
        assert compute_trend(series) == "INSUFFICIENT_DATA"

    def test_insufficient_data_empty(self) -> None:
        import pandas as pd

        from aer.utils.python_replacements import compute_trend

        assert compute_trend(pd.Series([], dtype=float)) == "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# format_indicator_summary
# ---------------------------------------------------------------------------


class TestFormatIndicatorSummary:
    def test_excludes_none(self) -> None:
        from aer.utils.python_replacements import format_indicator_summary

        result = format_indicator_summary({"RSI": 55.1, "MACD": None, "ADX": 28.3})
        assert "MACD" not in result
        assert "RSI: 55.1" in result
        assert "ADX: 28.3" in result

    def test_all_none_returns_empty_string(self) -> None:
        from aer.utils.python_replacements import format_indicator_summary

        assert format_indicator_summary({"A": None, "B": None}) == ""

    def test_pipe_separator(self) -> None:
        from aer.utils.python_replacements import format_indicator_summary

        result = format_indicator_summary({"X": 1, "Y": 2})
        assert " | " in result
