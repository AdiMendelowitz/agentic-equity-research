"""Equity market breadth indicators.

NYSE A/D tickers (^ADVN, ^DECL, ^ADD) were removed from Yahoo Finance.
Breadth score and sector dispersion are derived from SPDR sector ETFs only.
"""

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from aer.utils.cache import data_cache

logger = logging.getLogger(__name__)

_SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]


def get_breadth_snapshot() -> dict[str, Any]:
    """Market breadth snapshot via SPDR sector ETF dispersion. Cached 24 h.

    A/D line and McClellan Oscillator fields are always None (Yahoo removed ^ADVN/^DECL/^ADD).
    """
    key = "breadth:snapshot"
    cache = data_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    result: dict[str, Any] = {
        "ad_line_slope": None,
        "ad_line_trend": None,
        "mcclellan_oscillator": None,
        "mcclellan_signal": None,
        "net_advances_latest": None,
        "sector_dispersion": None,
        "breadth_score": None,
    }

    # NYSE A/D tickers removed from Yahoo Finance — A/D fields remain None

    # Sector dispersion: std of 20-day returns across SPDR sector ETFs
    try:
        sec_raw = yf.download(_SECTOR_ETFS, period="30d", interval="1d", progress=False, auto_adjust=True)
        if not sec_raw.empty:
            sec_close = sec_raw["Close"] if "Close" in sec_raw.columns else sec_raw.xs("Close", axis=1, level=0)
            if isinstance(sec_close, pd.Series):
                sec_close = sec_close.to_frame()
            sec_close = sec_close.dropna(how="all")
            if len(sec_close) >= 20:
                ret_20d = (sec_close.iloc[-1] / sec_close.iloc[-20] - 1).dropna()
                if not ret_20d.empty:
                    result["sector_dispersion"] = round(float(ret_20d.std()), 4)
    except Exception as e:
        logger.warning("Sector dispersion fetch failed: %s", e)

    # Composite breadth score (-1 to +1)
    score_components: list[float] = []
    osc = result.get("mcclellan_oscillator")
    if osc is not None:
        score_components.append(max(-1.0, min(1.0, osc / 150)))
    slope = result.get("ad_line_slope")
    if slope is not None:
        score_components.append(1.0 if slope > 0 else -1.0)
    if score_components:
        result["breadth_score"] = round(sum(score_components) / len(score_components), 3)

    cache.set(key, result)
    return result
