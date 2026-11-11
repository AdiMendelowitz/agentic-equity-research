"""Tests for rewired technical agent: _classify_signals, build_technical_prompt, run() metadata contract."""


class TestClassifySignals:
    """Unit tests for _classify_signals() — deterministic rule-based classification."""

    def test_bullish_scenario(self) -> None:
        from aer.agents.technical import _classify_signals

        ind = {
            "sma50": 110.0,
            "sma200": 100.0,
            "adx": 30.0,
            "rsi": 60.0,
            "stoch_k": 55.0,
            "roc": 1.5,
            "ao": 0.5,
            "price": 105.0,
            "bb_upper": 115.0,
            "bb_lower": 95.0,
            "atr": 2.1,
        }
        s = _classify_signals(ind)
        assert s["trend"] == "BULLISH"
        assert s["momentum"] == "BULLISH"
        assert s["overall"] == "BULLISH"

    def test_bearish_scenario(self) -> None:
        from aer.agents.technical import _classify_signals

        ind = {
            "sma50": 90.0,
            "sma200": 100.0,
            "adx": 30.0,
            "rsi": 40.0,
            "stoch_k": 25.0,
            "roc": -1.0,
            "ao": -0.5,
            "price": 92.0,
            "bb_upper": 110.0,
            "bb_lower": 80.0,
            "atr": 1.5,
        }
        s = _classify_signals(ind)
        assert s["trend"] == "BEARISH"
        assert s["momentum"] == "BEARISH"
        assert s["overall"] == "BEARISH"

    def test_empty_dict_no_keyerror(self) -> None:
        from aer.agents.technical import _classify_signals

        s = _classify_signals({})
        assert s["trend"] == "NEUTRAL"
        assert s["momentum"] == "NEUTRAL"
        assert s["mean_reversion"] == "NEUTRAL"
        assert s["volatility"] == "NORMAL"
        assert s["overall"] == "NEUTRAL"

    def test_atr_high_volatility(self) -> None:
        from aer.agents.technical import _classify_signals

        s = _classify_signals({"atr": 4.0, "price": 100.0})
        assert s["volatility"] == "HIGH"

    def test_atr_low_volatility(self) -> None:
        from aer.agents.technical import _classify_signals

        s = _classify_signals({"atr": 0.5, "price": 100.0})
        assert s["volatility"] == "LOW"

    def test_signals_dict_has_all_keys(self) -> None:
        from aer.agents.technical import _classify_signals

        s = _classify_signals({})
        assert set(s.keys()) == {"trend", "momentum", "mean_reversion", "volatility", "overall"}


class TestComputeIndicatorsNotInAgent:
    """Ensure _compute_indicators was deleted; compute_indicators is imported from indicators module."""

    def test_compute_indicators_deleted(self) -> None:
        import aer.agents.technical as m

        assert not hasattr(m, "_compute_indicators"), "_compute_indicators still present — delete it"

    def test_indicators_module_import_present(self) -> None:
        from pathlib import Path

        src_text = (Path(__file__).parent.parent / "src" / "aer" / "agents" / "technical.py").read_text()
        assert "from aer.indicators.technical import compute_indicators" in src_text

    def test_signals_in_metadata(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        import numpy as np
        import pandas as pd

        from aer.data.cache import Cache

        np.random.seed(0)
        n = 252
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
        df = pd.DataFrame(
            {
                "Open": close.shift(1).fillna(close.iloc[0]),
                "High": close + 0.3,
                "Low": close - 0.3,
                "Close": close,
                "Volume": pd.Series([1_000_000] * n, index=dates),
            }
        )
        mock_response = MagicMock()
        mock_response.content = "technical analysis"
        from aer.data.cache import TTL_ANALYSIS
        from aer.utils.cache import _DomainCache

        cache = Cache(db_path=str(tmp_path / "t.db"))
        with patch("aer.agents.technical.agent_cache", return_value=_DomainCache(cache, TTL_ANALYSIS, "agent")):
            with patch("aer.agents.technical.get_raw_history_last_year", return_value=df):
                with patch("aer.agents.technical.complete") as mock_complete:
                    mock_complete.return_value = mock_response
                    from aer.agents.technical import run

                    result = run("TEST")
        assert "signals" in result.metadata
        assert set(result.metadata["signals"].keys()) == {
            "trend",
            "momentum",
            "mean_reversion",
            "volatility",
            "overall",
        }


class TestBuildPrompt:
    """Unit tests for expanded _build_prompt() with SIGNAL DECOMPOSITION."""

    def _make_ind(self) -> dict:
        return {
            "price": 150.0,
            "rsi": 58.0,
            "macd": 1.23,
            "macd_signal": 1.10,
            "macd_hist": 0.13,
            "bb_upper": 155.0,
            "bb_mid": 150.0,
            "bb_lower": 145.0,
            "sma50": 148.0,
            "sma200": 140.0,
            "stoch_k": 65.0,
            "stoch_d": 60.0,
            "atr": 2.5,
            "obv": 5_500_000.0,
            "willr": -35.0,
            "cci": 85.0,
            "adx": 28.0,
            "adx_pos": 22.0,
            "adx_neg": 15.0,
            "ichi_tenkan": 149.0,
            "ichi_kijun": 147.0,
            "ichi_senkou_a": 146.0,
            "ichi_chikou": 151.0,
            "stochrsi_k": 70.0,
            "stochrsi_d": 65.0,
            "ao": 1.25,
            "mom": 3.0,
            "roc": 2.0,
            "vwma": 149.5,
            "ema_10": 151.0,
            "ema_20": 150.0,
            "ema_30": 149.0,
            "ema_50": 148.0,
            "ema_100": 145.0,
            "ema_200": 140.0,
            "vol_ratio": 1.15,
            "high_52w": 160.0,
            "low_52w": 120.0,
            "pct_from_high": -6.25,
            "pct_from_low": 25.0,
        }

    def test_signal_decomposition_section(self) -> None:
        from aer.agents.technical import _classify_signals
        from aer.utils.prompt_builders import build_technical_prompt

        ind = self._make_ind()
        signals = _classify_signals(ind)
        _, prompt = build_technical_prompt("TEST", ind, signals)
        assert "SIGNAL DECOMPOSITION" in prompt

    def test_buy_hold_sell_instruction(self) -> None:
        from aer.agents.technical import _classify_signals
        from aer.utils.prompt_builders import build_technical_prompt

        ind = self._make_ind()
        signals = _classify_signals(ind)
        _, prompt = build_technical_prompt("TEST", ind, signals)
        assert "Conclude with: BUY / HOLD / SELL" in prompt

    def test_new_indicator_labels_present(self) -> None:
        from aer.agents.technical import _classify_signals
        from aer.utils.prompt_builders import build_technical_prompt

        ind = self._make_ind()
        signals = _classify_signals(ind)
        _, prompt = build_technical_prompt("TEST", ind, signals)
        for label in ["ATR", "CCI", "ADX", "OBV", "AO", "Momentum", "ROC", "Ichimoku", "VWMA"]:
            assert label in prompt, f"Missing label: {label}"

    def test_obv_formatted_in_millions(self) -> None:
        from aer.agents.technical import _classify_signals
        from aer.utils.prompt_builders import build_technical_prompt

        ind = self._make_ind()
        signals = _classify_signals(ind)
        _, prompt = build_technical_prompt("TEST", ind, signals)
        assert "5.5m" in prompt, "OBV not formatted as millions"

    def test_three_param_signature(self) -> None:
        import inspect

        from aer.utils.prompt_builders import build_technical_prompt

        params = inspect.signature(build_technical_prompt).parameters
        assert list(params.keys()) == ["ticker", "indicators", "signals"]


class TestTechnicalRun:
    def test_uses_analysis_tier(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        import numpy as np
        import pandas as pd

        from aer.data.cache import Cache

        np.random.seed(42)
        n = 252
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = pd.Series(100.0 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
        df = pd.DataFrame(
            {
                "Open": close.shift(1).fillna(close.iloc[0]),
                "High": close + 0.3,
                "Low": close - 0.3,
                "Close": close,
                "Volume": pd.Series([1_000_000] * n, index=dates),
            }
        )
        mock_response = MagicMock()
        mock_response.content = "technical analysis"
        from aer.data.cache import TTL_ANALYSIS
        from aer.utils.cache import _DomainCache

        cache = Cache(db_path=str(tmp_path / "t.db"))
        with patch("aer.agents.technical.agent_cache", return_value=_DomainCache(cache, TTL_ANALYSIS, "agent")):
            with patch("aer.agents.technical.get_raw_history_last_year", return_value=df):
                with patch("aer.agents.technical.complete") as mock_complete:
                    mock_complete.return_value = mock_response
                    from aer.agents.technical import run

                    run("TEST")
        assert mock_complete.call_args.kwargs["tier"] == "ANALYSIS"
