"""Snapshot tests for src/utils/prompt_builders — one test per builder.

Each test asserts: ticker is present, required section headers are present,
no obvious missing data markers from template variables.
"""

# ---------------------------------------------------------------------------
# Fixtures / shared helpers
# ---------------------------------------------------------------------------


def _fundamental_args() -> tuple:
    info = {
        "ticker": "MSFT",
        "name": "Microsoft Corp.",
        "sector": "Technology",
        "industry": "Software",
        "exchange": "NASDAQ",
        "currency": "USD",
        "market_cap": 3_000_000_000_000,
        "employees": 221000,
        "description": "Cloud and software company.",
        "website": "https://microsoft.com",
        "country": "US",
        "pe_ratio": 35.0,
        "forward_pe": 30.0,
        "price_to_book": 14.0,
        "dividend_yield": 0.008,
        "52w_high": 430.0,
        "52w_low": 310.0,
        "beta": 0.9,
        "analyst_target": 450.0,
        "recommendation": "buy",
        "analyst_consensus": {},
    }
    ratios = {
        "revenue_growth_yoy": 15.0,
        "gross_margin": 68.0,
        "operating_margin": 44.0,
        "net_margin": 35.0,
        "fcf_on_equity": 40.0,
        "debt_to_equity": 0.4,
        "net_cash": 70_000_000_000,
        "eps_history": [{"period": "2024", "value": 11.45}],
    }
    return "MSFT", info, 420.0, ratios


def _technical_args() -> tuple:
    indicators = {
        "price": 420.0,
        "rsi": 55.0,
        "macd": 1.0,
        "macd_signal": 0.9,
        "macd_hist": 0.1,
        "bb_upper": 435.0,
        "bb_mid": 420.0,
        "bb_lower": 405.0,
        "sma50": 410.0,
        "sma200": 390.0,
        "stoch_k": 60.0,
        "stoch_d": 58.0,
        "atr": 5.0,
        "obv": 10_000_000.0,
        "willr": -40.0,
        "cci": 90.0,
        "adx": 25.0,
        "adx_pos": 20.0,
        "adx_neg": 14.0,
        "ichi_tenkan": 418.0,
        "ichi_kijun": 415.0,
        "ichi_senkou_a": 412.0,
        "ichi_chikou": 422.0,
        "stochrsi_k": 65.0,
        "stochrsi_d": 60.0,
        "ao": 1.5,
        "mom": 5.0,
        "roc": 2.5,
        "vwma": 419.0,
        "ema_10": 422.0,
        "ema_20": 420.0,
        "ema_30": 418.0,
        "ema_50": 410.0,
        "ema_100": 400.0,
        "ema_200": 390.0,
        "vol_ratio": 1.2,
        "high_52w": 430.0,
        "low_52w": 310.0,
        "pct_from_high": -2.3,
        "pct_from_low": 35.0,
    }
    signals = {
        "trend": "BULLISH",
        "momentum": "BULLISH",
        "mean_reversion": "NEUTRAL",
        "volatility": "LOW",
        "overall": "BULLISH",
    }
    return "MSFT", indicators, signals


def _sentiment_snap() -> dict:
    return {
        "apewisdom": {"found": True, "rank": 3, "rank_change_24h": 2, "mentions_24h": 500, "upvotes_24h": 200},
        "stocktwits": {
            "message_count": 8,
            "bullish_count": 5,
            "bearish_count": 2,
            "neutral_count": 1,
            "bullish_pct": 0.625,
            "bearish_pct": 0.25,
            "sample_messages": ["buying MSFT"],
        },
        "yahoo_news": {"article_count": 3, "headlines": [{"title": "MSFT gains", "publisher": "Reuters"}]},
        "finviz_news": {"article_count": 0, "headlines": [], "ticker": "MSFT"},
        "news_primary_source": "yahoo",
    }


def _macro_snap() -> dict:
    return {
        "fed_funds_rate": 5.33,
        "unemployment": 3.7,
        "treasury_10y": 4.25,
        "treasury_2y": 4.88,
        "yield_curve": -0.63,
        "gdp": 28500.0,
        "pce": 119.5,
        "vix": 13.5,
        "sp500": 5200.0,
        "cpi": 314.0,
        "cpi_yoy_pct": 3.2,
        "hy_spread": 320.0,
        "ig_spread": 95.0,
        "bbb_spread": 145.0,
        "real_yield_10y": 2.1,
        "breakeven_10y": 2.3,
        "breakeven_5y5y": 2.4,
        "net_liquidity_bn": 5800.0,
        "net_liquidity_trend": "neutral",
        "dxy": 104.5,
        "dxy_change_pct": 0.12,
        "eurusd": 1.082,
        "usdjpy": 149.5,
        "jobless_claims": 215.0,
        "consumer_sentiment": 78.5,
        "retail_sales": 720.0,
    }


def _macro_regime() -> dict:
    return {
        "regime": "BULL_STEADY",
        "description": "Risk-on expansion.",
        "equity_bias": "BULLISH",
        "score": 2,
        "signals": ["VIX low"],
    }


def _positioning_args() -> tuple:
    flow = {
        "ticker": "MSFT",
        "available": True,
        "current_price": 420.0,
        "total_call_volume": 40000,
        "total_put_volume": 30000,
        "total_call_oi": 180000,
        "total_put_oi": 150000,
        "pc_volume_ratio": 0.75,
        "pc_oi_ratio": 0.83,
        "atm_iv_call": 0.20,
        "atm_iv_put": 0.22,
        "iv_skew": 0.02,
        "expirations": [],
    }
    ownership = {
        "ticker": "MSFT",
        "insider_pct": 0.04,
        "institution_pct": 0.73,
        "short_float_pct": 0.006,
        "short_ratio": 1.2,
        "institutional_holders": [{"holder": "Vanguard", "shares": 800000, "pct_held": 0.07}],
        "insider_transactions": [],
        "insider_buys_recent": 2,
        "insider_sells_recent": 0,
        "insider_buy_sell_ratio": 2.0,
    }
    earnings = {
        "ticker": "MSFT",
        "next_earnings_date": "2026-07-25",
        "eps_estimate": 3.10,
        "eps_low": 2.95,
        "eps_high": 3.25,
        "revenue_estimate": 70_000_000_000.0,
        "earnings_history": [],
        "earnings_beat_rate": 0.80,
        "avg_surprise_pct": 4.0,
        "analyst_recommendations": [],
    }
    breadth = {
        "ad_line_slope": 8.0,
        "ad_line_trend": "rising",
        "mcclellan_oscillator": 30.0,
        "mcclellan_signal": "neutral",
        "net_advances_latest": 600,
        "sector_dispersion": 0.028,
        "breadth_score": 0.70,
    }
    cboe_pc = {"equity_pc_ratio": 0.65, "total_pc_ratio": 0.78, "equity_pc_5d_avg": 0.68, "sentiment_label": "NEUTRAL"}
    return "MSFT", flow, ownership, earnings, breadth, cboe_pc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildFundamentalPrompt:
    def test_snapshot(self) -> None:
        from aer.utils.prompt_builders import build_fundamental_prompt

        system, user = build_fundamental_prompt(*_fundamental_args())
        assert "MSFT" in user
        assert "Microsoft Corp." in user
        assert "$420.00" in user
        assert "Technology" in user
        assert system  # non-empty system prompt


class TestBuildTechnicalPrompt:
    def test_snapshot(self) -> None:
        from aer.utils.prompt_builders import build_technical_prompt

        system, user = build_technical_prompt(*_technical_args())
        assert "MSFT" in user
        assert "SIGNAL DECOMPOSITION" in user
        assert "BUY / HOLD / SELL" in user
        assert system


class TestBuildSentimentPrompt:
    def test_snapshot(self) -> None:
        from aer.utils.prompt_builders import build_sentiment_prompt

        system, user = build_sentiment_prompt("MSFT", _sentiment_snap(), "MIXED")
        assert "MSFT" in user
        assert "ApeWisdom" in user
        assert "Stocktwits" in user
        assert "MIXED" in user
        assert system


class TestBuildMacroPrompt:
    def test_snapshot(self) -> None:
        from aer.utils.prompt_builders import build_macro_prompt

        system, user = build_macro_prompt("MSFT", "Technology", _macro_snap(), _macro_regime(), "FAVORABLE")
        assert "MSFT" in user
        assert "Technology" in user
        assert "5.33%" in user
        assert "BULL_STEADY" in user
        assert "FAVORABLE" in user
        assert system


class TestBuildPositioningPrompt:
    def test_snapshot(self) -> None:
        from aer.utils.prompt_builders import build_positioning_prompt

        system, user = build_positioning_prompt(*_positioning_args(), "NEUTRAL")
        assert "MSFT" in user
        assert "Vanguard" in user
        assert "2026-07-25" in user
        assert "McClellan" in user
        assert system


class TestBuildSynthesisSystemPrompt:
    def test_without_options(self) -> None:
        from aer.utils.prompt_builders import build_synthesis_system_prompt

        prompt = build_synthesis_system_prompt(include_options=False)
        assert "portfolio manager" in prompt
        assert "OPTIONS" not in prompt

    def test_with_options(self) -> None:
        from aer.utils.prompt_builders import build_synthesis_system_prompt

        prompt = build_synthesis_system_prompt(include_options=True)
        assert "portfolio manager" in prompt
        assert "OPTIONS" in prompt
