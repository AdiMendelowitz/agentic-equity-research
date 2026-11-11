"""FRED macroeconomic data wrapper with SQLite caching.

Extended series coverage: credit spreads, real yields, inflation breakevens,
net Fed liquidity, FX snapshot, growth indicators, and regime classification.
"""

import logging
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import yfinance as yf
from fredapi import Fred

from aer.config import get_settings
from aer.utils.cache import data_cache
from aer.utils.constants import (
    NET_LIQUIDITY_DAILY_LOOKBACK_PERIODS,
    NET_LIQUIDITY_TREND_THRESHOLD_BN,
)
from aer.utils.vintage_tracker import record_vintage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED series registry
# ---------------------------------------------------------------------------

SERIES: dict[str, str] = {
    # Rates
    "fed_funds_rate": "FEDFUNDS",  # Effective Fed Funds Rate (monthly)
    "treasury_10y": "DGS10",  # 10-Year Treasury Yield (daily)
    "treasury_2y": "DGS2",  # 2-Year Treasury Yield (daily)
    "yield_curve": "T10Y2Y",  # 10Y-2Y spread (daily)
    # Real yields & inflation expectations
    "real_yield_5y": "DFII5",  # 5Y TIPS real yield (daily)
    "real_yield_10y": "DFII10",  # 10Y TIPS real yield (daily)
    "breakeven_10y": "T10YIE",  # 10Y inflation breakeven (daily)
    "breakeven_5y5y": "T5YIFR",  # 5Y5Y forward inflation expectation (daily)
    # Inflation
    "pce": "PCEPI",  # PCE Price Index (monthly)
    # Growth
    "gdp": "GDP",  # Real GDP (quarterly)
    "jobless_claims": "ICSA",  # Initial jobless claims (weekly)
    "consumer_sentiment": "UMCSENT",  # U of Michigan consumer sentiment (monthly)
    "retail_sales": "RSAFS",  # Advance retail sales (monthly)
    # Labor
    "unemployment": "UNRATE",  # Unemployment Rate (monthly)
    # Equity / volatility
    "vix": "VIXCLS",  # CBOE VIX (daily)
    "sp500": "SP500",  # S&P 500 Index (daily)
    # Credit spreads (BAML indices via FRED)
    "hy_spread": "BAMLH0A0HYM2",  # US HY OAS in basis points (daily)
    "ig_spread": "BAMLC0A0CM",  # US IG OAS in basis points (daily)
    "bbb_spread": "BAMLC0A4CBBB",  # US BBB OAS in basis points (daily)
    # Net Fed liquidity components (see get_net_liquidity())
    "fed_balance_sheet": "WALCL",  # Fed total assets, millions USD (weekly)
    "rrp": "RRPONTSYD",  # ON RRP outstanding, billions USD (daily)
    "tga": "WTREGEN",  # Treasury General Account, billions USD (weekly)
    # Money supply
    "m2": "M2SL",  # M2 money supply, billions USD (monthly)
    # Additional rates / credit / liquidity
    "treasury_3m": "DGS3MO",  # 3-Month Treasury Yield (daily)
    "treasury_5y": "DGS5",  # 5-Year Treasury Yield (daily)
    "treasury_30y": "DGS30",  # 30-Year Treasury Yield (daily)
    "sofr": "SOFR",  # Secured Overnight Financing Rate (daily)
    "aaa_spread": "BAMLC0A1CAAA",  # AAA corporate OAS in basis points (daily)
    # CPI All Urban Consumers (monthly) - raw index level; 'cpi' key owned by hardcoded snapshot block
    "cpi_raw": "CPIAUCSL",
}


def _get_fred() -> Fred:
    return Fred(api_key=get_settings().fred_api_key)


def get_series(
    series_id: str,
    periods: int = 24,
    *,
    realtime_start: date | str | None = None,
    realtime_end: date | str | None = None,
) -> pd.Series:
    """Return the last `periods` observations for a FRED series. Cached 24 h.

    When both `realtime_start` and `realtime_end` are None, behaviour and cache
    key are identical to the previous implementation (backward-compatible).

    When supplied, they are forwarded to ``fred.get_series`` for point-in-time
    vintage pulls; the cache key is extended with a cache_domain prefix so the
    vintaged series is stored separately from the live series. This prevents
    walk-forward data contamination.

    Parameters
    ----------
    realtime_start, realtime_end:
        ISO date string or datetime.date. Accepted as date objects and
        converted to ISO format strings for fredapi compatibility.
    """
    # Build cache key with domain prefix to separate live vs vintage
    if realtime_start is None and realtime_end is None:
        # Live data - no realtime params
        cache_domain = "macro:live"
        key = f"{cache_domain}:series:{series_id}:{periods}"
        realtime_kwargs: dict[str, str] = {}
    else:
        # Vintage data - point-in-time snapshot with separate cache domain
        cache_domain = "macro:vintage"
        rs_str = realtime_start.isoformat() if hasattr(realtime_start, "isoformat") else str(realtime_start)
        re_str = realtime_end.isoformat() if hasattr(realtime_end, "isoformat") else str(realtime_end)
        key = f"{cache_domain}:series:{series_id}:{periods}:{rs_str}:{re_str}"
        realtime_kwargs = {"realtime_start": rs_str, "realtime_end": re_str}

    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        return pd.Series(cached["values"], index=pd.to_datetime(cached["index"]), name=series_id)

    fred = _get_fred()
    series = fred.get_series(series_id, **realtime_kwargs)
    series = series.dropna().tail(periods)

    if series.empty:
        logger.warning("FRED series %s returned empty data — not caching", series_id)
        return series

    cache.set(
        key,
        {"values": series.tolist(), "index": series.index.astype(str).tolist()},
    )
    return series


def get_latest(series_id: str) -> float | None:
    """Return the most recent non-null value for a series."""
    try:
        s = get_series(series_id, periods=5)
        return float(s.iloc[-1]) if not s.empty else None
    except Exception as exc:
        logger.warning("get_latest(%s) failed: %s", series_id, exc)
        return None


def _align_net_liquidity(
    walcl_bn: pd.Series,
    rrp: pd.Series,
    tga: pd.Series,
    anchor: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Calendar-align net-liquidity components to a single anchor date via ``.asof``.

    WALCL (weekly), RRP (daily), and TGA (weekly) publish at different frequencies, so
    positional ``.iloc[-1]`` of each can refer to different calendar dates.
    ``anchor`` defaults to the latest WALCL date (the slowest-publishing series); RRP/TGA
    are aligned back to it, guaranteeing all three coexisted on/before one real
    balance-sheet date. ``anchor`` is a defaulted parameter only - no caller-facing
    ``as_of`` surface is exposed here.
    """
    if anchor is None:
        anchor = walcl_bn.index.max()

    # asof() raises ValueError on an unsorted index - guard the cache round-trip ordering.
    walcl_bn = walcl_bn.sort_index()
    rrp = rrp.sort_index()
    tga = tga.sort_index()

    bs_latest = walcl_bn.asof(anchor)
    rrp_latest = rrp.asof(anchor)
    tga_latest = tga.asof(anchor)

    if pd.isna(bs_latest) or pd.isna(rrp_latest) or pd.isna(tga_latest):
        return {
            "fed_balance_sheet_bn": None,
            "rrp_bn": None,
            "tga_bn": None,
            "net_liquidity_bn": None,
            "net_liquidity_trend": None,
        }

    net_latest = float(bs_latest) - float(rrp_latest) - float(tga_latest)

    # Trend: align the previous WALCL week's anchor; the daily RRP series is fetched wide
    # enough (NET_LIQUIDITY_DAILY_LOOKBACK_PERIODS) to reach prev_anchor on normal data.
    trend: str | None = None
    if len(walcl_bn) >= 2:
        prev_anchor = walcl_bn.index[-2]
        rrp_prev = rrp.asof(prev_anchor)
        tga_prev = tga.asof(prev_anchor)
        if not (pd.isna(rrp_prev) or pd.isna(tga_prev)):
            net_prev = float(walcl_bn.asof(prev_anchor)) - float(rrp_prev) - float(tga_prev)
            delta = net_latest - net_prev
            trend = (
                "expanding"
                if delta > NET_LIQUIDITY_TREND_THRESHOLD_BN
                else "contracting" if delta < -NET_LIQUIDITY_TREND_THRESHOLD_BN else "neutral"
            )

    return {
        "fed_balance_sheet_bn": round(float(bs_latest), 1),
        "rrp_bn": round(float(rrp_latest), 1),
        "tga_bn": round(float(tga_latest), 1),
        "net_liquidity_bn": round(net_latest, 1),
        "net_liquidity_trend": trend,
    }


def get_net_liquidity() -> dict[str, Any]:
    """Compute net Fed liquidity = Fed balance sheet - RRP - TGA.

    WALCL is in millions USD; RRPONTSYD and WTREGEN are in billions USD.
    All normalised to billions for the output.
    """
    key = "macro:net_liquidity"
    cache = data_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    result: dict[str, Any] = {
        "fed_balance_sheet_bn": None,
        "rrp_bn": None,
        "tga_bn": None,
        "net_liquidity_bn": None,
        "net_liquidity_trend": None,
    }

    try:
        # WALCL weekly, TGA weekly, RRP daily - fetch 4 weekly anchors (latest + prev) and
        # widen the daily RRP window so .asof(prev_anchor) reaches the prior WALCL week.
        walcl = get_series("WALCL", periods=4)
        rrp = get_series("RRPONTSYD", periods=NET_LIQUIDITY_DAILY_LOOKBACK_PERIODS)
        tga = get_series("WTREGEN", periods=4)

        walcl_bn = walcl / 1000  # millions → billions
        result = _align_net_liquidity(walcl_bn, rrp, tga)
    except Exception as exc:
        logger.warning("get_net_liquidity: FRED fetch failed: %s", exc)

    cache.set(key, result)
    return result


def get_fx_snapshot() -> dict[str, Any]:
    """DXY, EUR/USD, USD/JPY via yfinance (no API key required). Cached 24 h."""
    key = "macro:fx_snapshot"
    cache = data_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    result: dict[str, Any] = {
        "dxy": None,
        "dxy_change_pct": None,
        "eurusd": None,
        "eurusd_change_pct": None,
        "usdjpy": None,
        "usdjpy_change_pct": None,
    }

    _fx_tickers = {
        "dxy": "DX-Y.NYB",
        "eurusd": "EURUSD=X",
        "usdjpy": "JPY=X",
    }

    try:
        raw = yf.download(
            list(_fx_tickers.values()),
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            cache.set(key, result)
            return result

        close = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
        if isinstance(close, pd.Series):
            close = close.to_frame()
        close = close.dropna(how="all")

        for field, ticker in _fx_tickers.items():
            col = ticker
            if col not in close.columns:
                continue
            series = close[col].dropna()
            if len(series) >= 2:
                result[field] = round(float(series.iloc[-1]), 4)
                result[f"{field}_change_pct"] = round((series.iloc[-1] / series.iloc[-2] - 1) * 100, 3)
            elif len(series) == 1:
                result[field] = round(float(series.iloc[-1]), 4)
    except Exception as exc:
        logger.warning("get_fx_snapshot: yfinance fetch failed: %s", exc)

    cache.set(key, result)
    return result


def get_macro_snapshot() -> dict[str, Any]:
    """Unified macro snapshot: rates, spreads, real yields, FX, liquidity, growth. Cached 24 h."""
    key = "macro:snapshot"
    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        record_vintage("macro")
        return cached  # type: ignore[return-value]

    # Direct FRED fetches (all series except derived ones and net liquidity)
    _direct_series = {k: v for k, v in SERIES.items() if k not in ("fed_balance_sheet", "rrp", "tga", "cpi_raw")}
    snapshot: dict[str, Any] = {name: get_latest(sid) for name, sid in _direct_series.items()}

    # CPI with YoY
    try:
        cpi = get_series("CPIAUCSL", periods=15)
        snapshot["cpi"] = float(cpi.iloc[-1]) if not cpi.empty else None
        if len(cpi) >= 13:
            months_diff = (cpi.index[-1].year - cpi.index[-13].year) * 12 + (cpi.index[-1].month - cpi.index[-13].month)
            snapshot["cpi_yoy_pct"] = round((cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100, 2) if months_diff == 12 else None
        else:
            snapshot["cpi_yoy_pct"] = None
    except Exception as exc:
        logger.warning("get_macro_snapshot: CPI fetch failed: %s", exc)
        snapshot["cpi"] = None
        snapshot["cpi_yoy_pct"] = None

    # Net Fed liquidity
    liq = get_net_liquidity()
    snapshot["net_liquidity_bn"] = liq.get("net_liquidity_bn")
    snapshot["net_liquidity_trend"] = liq.get("net_liquidity_trend")

    # FX
    fx = get_fx_snapshot()
    snapshot.update(fx)

    snapshot["fetched_at"] = datetime.now(UTC).isoformat()
    cache.set(key, snapshot)
    record_vintage("macro")
    return snapshot
