"""yfinance wrapper with SQLite caching. Analyst targets cross-validated with FinViz."""

import logging
import math
from datetime import date
from typing import Any

import pandas as pd
import yfinance as yf

from aer.utils.cache import TTL_QUOTE, data_cache
from aer.utils.vintage_tracker import record_vintage

logger = logging.getLogger(__name__)


_RECOM_LABELS = {
    1: "Strong Buy",
    2: "Buy",
    3: "Hold",
    4: "Sell",
    5: "Strong Sell",
}


def _clean_value(v: Any) -> Any:
    """Convert numpy scalars to Python-native; replace NaN with None."""
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def _df_to_cacheable(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": df.columns.tolist(),
        "index": df.index.astype(str).tolist(),
        "data": [[_clean_value(v) for v in row] for row in df.values.tolist()],
    }


def _cacheable_to_df(cached: dict[str, Any]) -> pd.DataFrame:
    # utc=True parses offset strings that span a DST change (-04:00 and -05:00 in one index, which
    # a plain to_datetime leaves as an object index with no .tz), then drops the zone: the same
    # UTC-naive result the tz_convert(None) path always produced, and naive strings pass through
    # unchanged. The cold path now returns through here, so a 1y aware frame must parse.
    _idx = pd.to_datetime(cached["index"], utc=True).tz_convert(None)
    df = pd.DataFrame(cached["data"], columns=cached["columns"], index=_idx)
    df.index.name = "Date"
    return df


def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """OHLCV history. Cached for TTL_PRICES seconds."""
    ticker = ticker.upper()
    key = f"prices:ohlcv:{ticker}:{period}:{interval}"
    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        df = _cacheable_to_df(cached)
        record_vintage("prices")
        return df

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        record_vintage("prices")
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(how="all")
    if df.empty:
        record_vintage("prices")
        return df

    payload = _df_to_cacheable(df)
    cache.set(key, payload)
    # return the same round-trip the warm path reads, so cold == warm by construction
    # (tz-naive index, None-for-NaN normalisation) and no caller needs an inline tz patch.
    df = _cacheable_to_df(payload)
    record_vintage("prices")
    return df


def get_price_history_raw(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """Dividend-unadjusted OHLCV for backtesting.

    Uses auto_adjust=False. NOTE (verified yfinance 1.3.0): this does NOT leave splits
    unadjusted — yfinance always applies stock-split adjustments, so the returned Close is
    split-adjusted (continuous across splits, no artificial 4x jump) but dividend-UNADJUSTED.
    That is exactly what signal computation wants: split continuity avoids spurious
    indicator spikes, while skipping dividend adjustment avoids the look-ahead of
    retroactively rescaling every historical bar each time a later dividend is paid.
    The adjusted path (get_price_history, auto_adjust=True) additionally folds in dividends.
    """
    ticker = ticker.upper()
    key = f"prices:ohlcv_raw:{ticker}:{period}:{interval}"
    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        df = _cacheable_to_df(cached)
        return df

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(how="all")
    if df.empty:
        return df

    payload = _df_to_cacheable(df)
    cache.set(key, payload)
    df = _cacheable_to_df(payload)  # cold == warm, see get_price_history
    return df


def get_raw_history_last_year(ticker: str) -> pd.DataFrame:
    """Dividend-unadjusted daily bars for the last 365 days signal path).

    Reads the prefetch-warmed ``prices:ohlcv_raw:{ticker}:3y:1d`` key and slices it, so moving the
    ATR / technical computations onto the raw path costs no extra network call per pipeline run.
    """
    df = get_price_history_raw(ticker)
    if df.empty:
        return df
    return df[df.index > df.index[-1] - pd.DateOffset(years=1)]


def get_raw_history_since_uncached(ticker: str, start: date) -> pd.DataFrame:
    """Daily bars from `start` with BOTH Close and Adj Close, read straight from the source.

    Used by the simulator's persisted benchmark, which chains consecutive bars and must see
    the latest completed session on every run: a cached frame keyed on a repeating period would
    hand back yesterday's bars inside the 24 h TTL. No cache read, no cache write.
    """
    df = yf.download(ticker.upper(), start=start.isoformat(), interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def get_current_price(ticker: str) -> float | None:
    """Latest closing price. Falls back to fast_info if history is unavailable."""
    ticker = ticker.upper()
    key = f"prices:current:{ticker}"
    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        return float(cached)

    tk = yf.Ticker(ticker)
    price: float | None = None

    # Distinguish expected exceptions (data missing) from errors
    try:
        price = tk.fast_info.last_price
    except (AttributeError, KeyError):
        # yfinance fast_info not available for this ticker — expected case
        pass
    except Exception as exc:
        # Unexpected exception — log for debugging
        logger.debug("yfinance fast_info failed for %s: %s", ticker, exc)

    if price is None:
        hist = get_price_history(ticker, period="5d", interval="1d")
        if not hist.empty and "Close" in hist.columns:
            close = hist["Close"].dropna()  # skip pre-market forming rows with NaN Close
            if not close.empty:
                price = float(close.iloc[-1])

    # Both fast_info.last_price and a trailing forming row can yield NaN — treat as unavailable
    # so a NaN is never cached or returned (mirrors simulator._get_current_price).
    if price is not None and math.isnan(price):
        price = None

    if price is not None:
        cache.set(key, price, ttl=TTL_QUOTE)

    return price


def get_analyst_consensus(ticker: str) -> dict[str, Any]:
    """Cross-validate analyst price targets between yfinance and FinViz."""
    ticker = ticker.upper()
    key = f"prices:analyst_consensus:{ticker}"
    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        record_vintage("consensus")
        return cached  # type: ignore[return-value]

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        yf_target: float | None = info.get("targetMeanPrice")
        yf_recom: str = info.get("recommendationKey", "")

        fvz_target: float | None = None
        fvz_recom_score: float | None = None
        discrepancies: list[str] = []

        logger.debug("FinViz cross-validation unavailable for %s", ticker)

        if yf_target is not None and fvz_target is not None and yf_target != 0:
            diff_pct = (fvz_target / yf_target - 1) * 100
            if abs(diff_pct) > 10:
                discrepancies.append(
                    f"Price Target: yf=${yf_target:.2f} vs finviz=${fvz_target:.2f} ({diff_pct:+.1f}%)"
                )

        available = [t for t in (yf_target, fvz_target) if t is not None]
        consensus_target = round(sum(available) / len(available), 2) if available else None

        fvz_recom_label: str | None = None
        if fvz_recom_score is not None:
            nearest = min(_RECOM_LABELS.keys(), key=lambda k: abs(k - fvz_recom_score))
            fvz_recom_label = _RECOM_LABELS[nearest]

        result: dict[str, Any] = {
            "ticker": ticker,
            "yf_target": yf_target,
            "finviz_target": fvz_target,
            "consensus_target": consensus_target,
            "yf_recommendation": yf_recom,
            "finviz_recom_score": fvz_recom_score,
            "finviz_recom_label": fvz_recom_label,
            "_discrepancies": discrepancies,
        }
        cache.set(key, result)
        record_vintage("consensus")
        return result

    except Exception as e:
        logger.warning("Analyst consensus failed for %s: %s", ticker, e)
        record_vintage("consensus")
        return {"ticker": ticker}


def get_company_metadata(ticker: str) -> dict[str, Any]:
    """Lean company metadata: name, sector, exchange, currency — no analyst lookup.

    Use when analyst consensus is not needed (e.g. macro agent sector lookup).
    """
    ticker = ticker.upper()
    key = f"prices:metadata:{ticker}"
    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    tk = yf.Ticker(ticker)
    raw: dict[str, Any] = tk.info or {}

    info = {
        "ticker": ticker,
        "name": raw.get("longName") or raw.get("shortName", ticker),
        "sector": raw.get("sector", ""),
        "industry": raw.get("industry", ""),
        "exchange": raw.get("exchange", ""),
        "currency": raw.get("currency", "USD"),
        "country": raw.get("country", ""),
        "market_cap": raw.get("marketCap"),
        "description": raw.get("longBusinessSummary", ""),
    }

    cache.set(key, info)
    return info


def get_company_info(ticker: str) -> dict[str, Any]:
    """Company metadata: name, sector, exchange, market cap, description, etc."""
    ticker = ticker.upper()
    key = f"prices:info:{ticker}"
    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    tk = yf.Ticker(ticker)
    raw: dict[str, Any] = tk.info or {}

    consensus = get_analyst_consensus(ticker)
    consensus_target = consensus.get("consensus_target")

    info = {
        "ticker": ticker,
        "name": raw.get("longName") or raw.get("shortName", ticker),
        "sector": raw.get("sector", ""),
        "industry": raw.get("industry", ""),
        "exchange": raw.get("exchange", ""),
        "currency": raw.get("currency", "USD"),
        "market_cap": raw.get("marketCap"),
        "employees": raw.get("fullTimeEmployees"),
        "description": raw.get("longBusinessSummary", ""),
        "website": raw.get("website", ""),
        "country": raw.get("country", ""),
        "pe_ratio": raw.get("trailingPE"),
        "forward_pe": raw.get("forwardPE"),
        "price_to_book": raw.get("priceToBook"),
        "dividend_yield": raw.get("dividendYield"),
        "52w_high": raw.get("fiftyTwoWeekHigh"),
        "52w_low": raw.get("fiftyTwoWeekLow"),
        "beta": raw.get("beta"),
        "analyst_target": consensus_target if consensus_target is not None else raw.get("targetMeanPrice"),
        "recommendation": raw.get("recommendationKey", ""),
        "analyst_consensus": consensus,
    }

    cache.set(key, info)
    return info
