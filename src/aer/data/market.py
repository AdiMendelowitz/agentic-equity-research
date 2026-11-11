"""Batch market snapshot data: indices, sector ETFs, crypto. Cached 15 minutes."""

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from aer.utils.cache import data_cache

logger = logging.getLogger(__name__)

TTL_MARKET: int = 900  # 15 minutes

INDICES: dict[str, str] = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "Russell 2000": "^RUT",
}

SECTOR_ETFS: dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Disc.": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communications": "XLC",
}

CRYPTO_TICKERS: list[str] = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"]


def get_batch_snapshot(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """
    Batch fetch last 5 trading days for each ticker.
    Returns {ticker: {price, change_pct, change_abs}}.
    Cached for TTL_MARKET seconds (15 min).
    """
    if not tickers:
        return {}

    key = f"market:batch:{','.join(sorted(set(tickers)))}"
    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    result: dict[str, dict[str, Any]] = {}

    try:
        fetch_arg: Any = tickers if len(tickers) > 1 else tickers[0]
        raw = yf.download(fetch_arg, period="5d", interval="1d", progress=False, auto_adjust=True)
        if raw.empty:
            return {}

        is_multi = isinstance(raw.columns, pd.MultiIndex)
        if is_multi:
            raw.columns = pd.MultiIndex.from_tuples([(f, s.upper()) for f, s in raw.columns], names=raw.columns.names)

        for ticker in tickers:
            try:
                close: pd.Series = raw["Close"][ticker.upper()].dropna() if is_multi else raw["Close"].dropna()
                if len(close) < 2:
                    continue
                current = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                change_pct = (current / prev - 1) * 100
                result[ticker] = {
                    "price": current,
                    "change_pct": change_pct,
                    "change_abs": current - prev,
                }
            except (KeyError, IndexError, TypeError):
                continue

    except Exception as e:
        logger.warning("Batch snapshot failed for %s: %s", tickers, e)
        return {}

    if result:
        cache.set(key, result, ttl=TTL_MARKET)
    return result
