"""Technical indicator computation module."""

from typing import Any

import pandas as pd
import pandas_ta as ta  # type: ignore  # noqa: F401 — registers df.ta accessor


def compute_indicators(df: pd.DataFrame) -> dict[str, Any]:
    """Pure computation — no LLM, no cache, no side effects. Returns flat dict of indicator values."""
    close = df["Close"]
    volume = df["Volume"]
    latest = float(close.iloc[-1])

    def last(col: str, src: pd.DataFrame | None = None) -> float | None:
        frame = src if src is not None else df
        s = frame[col].dropna() if col in frame.columns else pd.Series(dtype=float)
        return float(s.iloc[-1]) if not s.empty else None

    # --- existing indicators ---
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)

    # --- new P1 indicators ---
    df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
    df.ta.atr(length=14, append=True)  # ATRr_14 = absolute $ value, not percent
    df.ta.obv(append=True)
    df.ta.willr(length=14, append=True)
    df.ta.cci(length=20, append=True)  # key is "CCI_20_0.015" including the Lambert constant
    df.ta.adx(length=14, append=True)
    df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3, append=True)
    df.ta.ao(append=True)
    df.ta.mom(length=10, append=True)
    df.ta.roc(append=True)
    df.ta.vwma(length=20, append=True)

    # Ichimoku: returns (ichi_df, ichi_cloud_df) and mutates df even without append=True.
    # Read values from ichi directly to avoid duplicate-column issues after the implicit mutation.
    # _ichi_cloud excluded: forward-shifted 26 periods = look-ahead bias
    # LOOK-AHEAD RISK: ISA_9 (Senkou Span A) may be forward-shifted 26 periods in some pandas-ta
    # versions. Only ichi.iloc[-27] is guaranteed look-ahead-free; ichi.iloc[-1] may use bars N+1
    # through N+26. Acceptable for current use (end-of-day signal only, not intrabar).
    ichi, _ichi_cloud = df.ta.ichimoku(tenkan=9, kijun=26, senkou=52, include_chikou=True)

    # EMA loop: guard len(df) >= ema_len — pandas-ta 0.4.71b0 returns full DataFrame silently if len < length
    for ema_len in [10, 20, 30, 50, 100, 200]:
        if len(df) >= ema_len:
            df.ta.ema(length=ema_len, append=True)

    # derived stats
    vol_20d_avg = float(volume.tail(20).mean()) if len(volume) >= 20 else None
    vol_ratio = float(volume.iloc[-1] / vol_20d_avg) if vol_20d_avg else None
    high_52w = float(close.tail(252).max()) if len(close) >= 252 else None
    low_52w = float(close.tail(252).min()) if len(close) >= 252 else None

    return {
        "price": latest,
        # existing (BBands key format: BBL_20_2.0_2.0 in pandas-ta >= 0.3.14b)
        "rsi": last("RSI_14"),
        "macd": last("MACD_12_26_9"),
        "macd_signal": last("MACDs_12_26_9"),
        "macd_hist": last("MACDh_12_26_9"),
        "bb_upper": last("BBU_20_2.0_2.0"),
        "bb_mid": last("BBM_20_2.0_2.0"),
        "bb_lower": last("BBL_20_2.0_2.0"),
        "sma50": last("SMA_50"),
        "sma200": last("SMA_200"),
        # new P1
        "stoch_k": last("STOCHk_14_3_3"),
        "stoch_d": last("STOCHd_14_3_3"),
        "atr": last("ATRr_14"),
        "obv": last("OBV"),
        "willr": last("WILLR_14"),
        "cci": last("CCI_20_0.015"),
        "adx": last("ADX_14"),
        "adx_pos": last("DMP_14"),
        "adx_neg": last("DMN_14"),
        # Ichimoku: read from ichi DataFrame directly to avoid duplicate-column confusion
        "ichi_tenkan": last("ITS_9", src=ichi),
        "ichi_kijun": last("IKS_26", src=ichi),
        "ichi_senkou_a": last("ISA_9", src=ichi),
        "ichi_chikou": last("ICS_26", src=ichi),
        "stochrsi_k": last("STOCHRSIk_14_14_3_3"),
        "stochrsi_d": last("STOCHRSId_14_14_3_3"),
        "ao": last("AO_5_34"),
        "mom": last("MOM_10"),
        "roc": last("ROC_10"),
        "vwma": last("VWMA_20"),
        "ema_10": last("EMA_10"),
        "ema_20": last("EMA_20"),
        "ema_30": last("EMA_30"),
        "ema_50": last("EMA_50"),
        "ema_100": last("EMA_100"),
        "ema_200": last("EMA_200"),
        # derived
        "vol_ratio": vol_ratio,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_from_high": ((latest / high_52w) - 1) * 100 if high_52w else None,
        "pct_from_low": ((latest / low_52w) - 1) * 100 if low_52w else None,
    }
