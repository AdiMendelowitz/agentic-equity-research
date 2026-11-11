"""Earnings calendar, EPS estimates, and historical surprises from yfinance."""

import logging
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
import yfinance as yf

from aer.utils.cache import TTL_QUOTE, agent_cache
from aer.utils.vintage_tracker import record_vintage

logger = logging.getLogger(__name__)

# CF-12: catalyst rows older than this window lose to a present yfinance date.
_CATALYST_EARNINGS_MAX_AGE_DAYS: int = 30

# surprise% = (reported - estimated)/abs(estimated)*100 explodes when the estimate
# base is near zero (a penny miss on a $0.01 estimate reads as a 100% surprise and swamps
# the average). Skip the surprise computation below this absolute EPS base. Surprise% is
# inherently unstable for low-EPS names; median_surprise_pct is the robust companion stat.
_MIN_EPS_BASE_FOR_SURPRISE: float = 0.05

# A-02/T-09: the resolved catalyst override date is cached under this short-TTL key so the
# hot path (a cache hit on the full earnings dict) does not open a fresh SQLite connection
# and re-run the catalyst query on every call. TTL_QUOTE (1h) is the shortest existing
# sensible TTL - short enough that a same-day reschedule is picked up promptly.
_NEXT_OVERRIDE_TTL: int = TTL_QUOTE


def _today_utc() -> date:
    return datetime.now(tz=UTC).date()


def _safe_int(val: Any) -> int:
    try:
        return int(val) if pd.notna(val) else 0
    except Exception:
        return 0


def _coerce_finite_float(val: Any) -> float | None:
    """Coerce a yfinance cell to a finite float, or None.

    T-03: pd.isna can raise on array-valued cells and float() raises on non-scalars.
    Treat anything that does not cleanly coerce to a finite float as None.
    """
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN or inf
        return None
    return f


def _catalyst_earnings_override(ticker: str, yf_date_str: str | None) -> str | None:
    """Catalyst table is excluded: returns the yfinance date unchanged."""
    return yf_date_str


def _resolve_next_earnings_override(ticker: str, yf_date_str: str | None) -> str | None:
    """Resolve the catalyst override once and cache the scalar result (A-02/T-09).

    On a short-TTL hit, return the cached resolved date without opening a SQLite connection.
    Only re-query the catalyst table when the key is absent/expired.
    """
    cache = agent_cache()
    override_key = f"earnings_next:{ticker}"
    if override_key in cache:
        cached_date = cache.get(override_key)
        # A stored None (no override / no yfinance date) is a legitimate resolved value.
        return cached_date if isinstance(cached_date, str) else None
    resolved = _catalyst_earnings_override(ticker, yf_date_str)
    cache.set(override_key, resolved, ttl=_NEXT_OVERRIDE_TTL)
    return resolved


def get_next_earnings_date(ticker: str) -> date | None:
    """Return the next earnings date for ticker, with catalyst table override (CF-12).

    Checks the catalyst_events table first (SKILL/LSEG sources take precedence).
    Falls back to yfinance calendar on miss, table absence, or fresh install.
    """
    data = get_earnings_data(ticker)
    date_str = data.get("next_earnings_date")
    if date_str is None:
        return None
    # T-22: a non-ISO yfinance string (e.g. "06/01/2026") must not crash or silently yield
    # None via fromisoformat. pd.to_datetime(errors="coerce") parses common formats and
    # returns NaT on genuine garbage.
    parsed = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _extract_calendar_dates(cal: Any) -> list[Any]:
    """Collect raw earnings-date candidates from a yfinance calendar (dict or DataFrame).

    T-03: the dict branch's value may be a scalar (incl. NaT), an empty list, or a list with
    NaT entries; the DataFrame branch exposes candidates on its index. Returns the raw cells;
    NaT/None filtering and forward selection happen in the caller.
    """
    if isinstance(cal, dict):
        dates = cal.get("Earnings Date", [])
        if dates is None:
            return []
        if isinstance(dates, (list, tuple)):
            return list(dates)
        return [dates]  # scalar (Timestamp, NaT, str, ...)
    if isinstance(cal, pd.DataFrame) and not cal.empty:
        return list(cal.index)
    return []


def _pick_next_forward_date(candidates: list[Any], today: date) -> str | None:
    """Return the earliest candidate date >= today as ISO string, or None.

    yfinance returns the last-reported date as the first calendar entry, so taking
    dates[0]/index[0] verbatim can surface a past date as "next earnings". Coerce each
    candidate (guarding NaT), then pick the smallest forward date. None when nothing is
    forward - never a stale past date.
    """
    forward: list[date] = []
    for it in candidates:
        if not pd.notna(it):  # guards NaT/None before .date()
            continue
        try:
            ts = pd.to_datetime(it, errors="coerce")
        except (TypeError, ValueError):
            continue
        if pd.isna(ts):
            continue
        d = ts.date()
        if d >= today:
            forward.append(d)
    if not forward:
        return None
    return min(forward).isoformat()


def get_earnings_data(ticker: str) -> dict[str, Any]:
    """Next earnings date, EPS estimates, historical surprises, analyst recommendations."""
    ticker = ticker.upper()
    key = f"earnings:{ticker}"
    cache = agent_cache()

    cached = cache.get(key)
    if cached is not None:
        # A-02/T-09: build the returned dict by shallow-copying the cached dict and replacing
        # only the next_earnings_date scalar. The override path mutates nothing else - the
        # nested earnings_history/analyst_recommendations lists are not touched here, and no
        # caller mutates them in place (they are read-only render inputs), so a shallow copy
        # is safe and avoids a per-call deepcopy of the whole tree.
        result: dict[str, Any] = dict(cached)  # type: ignore[arg-type]
        result["next_earnings_date"] = _resolve_next_earnings_override(ticker, result.get("next_earnings_date"))
        record_vintage("earnings")
        return result

    today_utc = _today_utc()
    result = {"ticker": ticker}
    errors: list[str] = []

    try:
        tk = yf.Ticker(ticker)

        # Next earnings date + consensus estimates
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                result["next_earnings_date"] = _pick_next_forward_date(_extract_calendar_dates(cal), today_utc)
                result["eps_estimate"] = cal.get("Earnings Average")
                result["eps_low"] = cal.get("Earnings Low")
                result["eps_high"] = cal.get("Earnings High")
                result["revenue_estimate"] = cal.get("Revenue Average")
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                result["next_earnings_date"] = _pick_next_forward_date(_extract_calendar_dates(cal), today_utc)
        except Exception as e:
            logger.warning("Earnings calendar failed for %s: %s", ticker, e)
            errors.append(f"calendar: {type(e).__name__}")

        # Historical earnings with beat/miss
        try:
            ed = tk.earnings_dates
            if ed is not None and not ed.empty:
                history: list[dict[str, Any]] = []
                for dt, row in ed.head(8).iterrows():
                    reported_f = _coerce_finite_float(row.get("Reported EPS"))
                    estimated_f = _coerce_finite_float(row.get("EPS Estimate"))
                    surprise_pct = None
                    # skip near-zero estimate bases so a penny-base quarter cannot
                    # swamp the average; such rows still appear in history with None surprise.
                    if (
                        reported_f is not None
                        and estimated_f is not None
                        and abs(estimated_f) >= _MIN_EPS_BASE_FOR_SURPRISE
                    ):
                        surprise_pct = round((reported_f - estimated_f) / abs(estimated_f) * 100, 2)
                    row_date = dt.date() if hasattr(dt, "date") else None
                    history.append(
                        {
                            "date": row_date.isoformat() if row_date is not None else str(dt),
                            "_date_obj": row_date,
                            "reported_eps": reported_f,
                            "estimated_eps": estimated_f,
                            "surprise_pct": surprise_pct,
                        }
                    )

                # as-of guard - a beat-rate must reflect reported quarters only. Exclude
                # rows dated today or in the future explicitly (the unreported quarter), rather
                # than relying on a null surprise coinciding with it.
                completed = [
                    s
                    for s in history
                    if s["surprise_pct"] is not None and s["_date_obj"] is not None and s["_date_obj"] < today_utc
                ]
                for s in history:
                    del s["_date_obj"]

                result["earnings_history"] = history
                # surface sample size so a "100% beat rate" from n=1 is visible.
                result["earnings_sample_size"] = len(completed)
                if completed:
                    surprises = [s["surprise_pct"] for s in completed]
                    beats = sum(1 for v in surprises if v > 0)
                    result["earnings_beat_rate"] = round(beats / len(completed), 2)
                    result["avg_surprise_pct"] = round(sum(surprises) / len(surprises), 2)
                    # median is robust to the low-EPS outliers that distort the mean.
                    result["median_surprise_pct"] = round(_median(surprises), 2)
        except Exception as e:
            logger.warning("Earnings history failed for %s: %s", ticker, e)
            errors.append(f"history: {type(e).__name__}")

        # Analyst recommendation trend
        try:
            recs = tk.recommendations
            if recs is not None and not recs.empty:
                result["analyst_recommendations"] = [
                    {
                        "period": str(idx),
                        "strong_buy": _safe_int(row.get("strongBuy")),
                        "buy": _safe_int(row.get("buy")),
                        "hold": _safe_int(row.get("hold")),
                        "sell": _safe_int(row.get("sell")),
                        "strong_sell": _safe_int(row.get("strongSell")),
                    }
                    for idx, row in recs.tail(4).iterrows()
                ]
        except Exception as e:
            logger.warning("Analyst recommendations failed for %s: %s", ticker, e)
            errors.append(f"recommendations: {type(e).__name__}")

        if errors:
            result["_errors"] = errors
        cache.set(key, result)
    except Exception as e:
        logger.warning("Earnings data failed for %s: %s", ticker, e)
        errors.append(f"ticker: {type(e).__name__}")
        result["_errors"] = errors

    # CF-12: apply the catalyst override at the cache-miss path (after yfinance fetch),
    # routing through the cached resolver so the hot path stays connection-free (A-02).
    result["next_earnings_date"] = _resolve_next_earnings_override(ticker, result.get("next_earnings_date"))
    record_vintage("earnings")
    return result


def _median(values: list[float]) -> float:
    """Median of a non-empty numeric list (avoids a statistics import for one call site)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
