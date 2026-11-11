"""Options chain analysis from yfinance — put/call ratios, IV, skew."""

import logging
from typing import Any, NamedTuple

import pandas as pd
import yfinance as yf

from aer.utils.cache import TTL_ANALYSIS, agent_cache, market_aware_ttl
from aer.utils.vintage_tracker import record_vintage

logger = logging.getLogger(__name__)


class OptionChain(NamedTuple):
    calls: pd.DataFrame
    puts: pd.DataFrame


def _df_to_records(df: pd.DataFrame) -> dict[str, Any]:
    """Serialize DataFrame to JSON-safe dict; NaN replaced with None."""
    return {
        "columns": df.columns.tolist(),
        "records": df.where(pd.notnull(df), other=None).to_dict(orient="records"),
    }


def _records_to_df(cached: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(cached["records"], columns=cached["columns"])


def get_option_chain(ticker: str, expiry: str) -> OptionChain | None:
    """Return cached option chain (calls, puts) for ticker/expiry. Cached at TTL_ANALYSIS.

    Returns None if the chain cannot be fetched so callers can skip gracefully.
    """
    ticker = ticker.upper()
    key = f"options:chain:{ticker}:{expiry}"
    cache = agent_cache()

    cached = cache.get(key)
    if cached is not None:
        return OptionChain(calls=_records_to_df(cached["calls"]), puts=_records_to_df(cached["puts"]))

    try:
        raw = yf.Ticker(ticker).option_chain(expiry)
        chain = OptionChain(calls=raw.calls, puts=raw.puts)
        cache.set(
            key,
            {"calls": _df_to_records(chain.calls), "puts": _df_to_records(chain.puts)},
            ttl=market_aware_ttl(TTL_ANALYSIS),
        )
        return chain
    except Exception as exc:
        logger.warning("get_option_chain: fetch failed for %s expiry %s: %s", ticker, expiry, exc)
        return None


def get_options_flow(ticker: str, num_expirations: int = 3) -> dict[str, Any]:
    """Aggregate put/call volume/OI ratios, ATM IV, and IV skew across nearest expirations."""
    ticker = ticker.upper()
    key = f"options:flow:{ticker}:{num_expirations}"
    cache = agent_cache()

    cached = cache.get(key)
    if cached is not None:
        record_vintage("options")
        return cached  # type: ignore[return-value]

    unavailable: dict[str, Any] = {"ticker": ticker, "available": False}

    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options

        if not expirations:
            return unavailable

        current_price: float | None = None
        try:
            current_price = tk.fast_info.last_price
        except Exception:
            pass

        total_call_vol = total_put_vol = total_call_oi = total_put_oi = 0
        atm_iv_calls: list[float] = []
        atm_iv_puts: list[float] = []
        expiry_summaries: list[dict[str, Any]] = []

        for exp in expirations[:num_expirations]:
            try:
                chain = tk.option_chain(exp)
                calls, puts = chain.calls, chain.puts

                c_vol = int(calls["volume"].fillna(0).sum()) if "volume" in calls else 0
                p_vol = int(puts["volume"].fillna(0).sum()) if "volume" in puts else 0
                c_oi = int(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls else 0
                p_oi = int(puts["openInterest"].fillna(0).sum()) if "openInterest" in puts else 0

                total_call_vol += c_vol
                total_put_vol += p_vol
                total_call_oi += c_oi
                total_put_oi += p_oi

                if current_price is not None and "impliedVolatility" in calls.columns:
                    if calls["strike"].notna().any():
                        atm_c = calls.loc[[calls["strike"].sub(current_price).abs().idxmin()]]
                        iv_c = atm_c["impliedVolatility"].iloc[0] if not atm_c.empty else None
                        if iv_c is not None and not pd.isna(iv_c):
                            atm_iv_calls.append(float(iv_c))
                    if puts["strike"].notna().any():
                        atm_p = puts.loc[[puts["strike"].sub(current_price).abs().idxmin()]]
                        iv_p = atm_p["impliedVolatility"].iloc[0] if not atm_p.empty else None
                        if iv_p is not None and not pd.isna(iv_p):
                            atm_iv_puts.append(float(iv_p))

                expiry_summaries.append(
                    {
                        "expiration": exp,
                        "call_volume": c_vol,
                        "put_volume": p_vol,
                        "call_oi": c_oi,
                        "put_oi": p_oi,
                        "pc_volume_ratio": round(p_vol / c_vol, 3) if c_vol else None,
                    }
                )
            except Exception as e:
                logger.warning("Options chain failed for %s exp %s: %s", ticker, exp, e)

        atm_iv_call = round(sum(atm_iv_calls) / len(atm_iv_calls), 4) if atm_iv_calls else None
        atm_iv_put = round(sum(atm_iv_puts) / len(atm_iv_puts), 4) if atm_iv_puts else None
        iv_skew = round(atm_iv_put - atm_iv_call, 4) if (atm_iv_put is not None and atm_iv_call is not None) else None

        result: dict[str, Any] = {
            "ticker": ticker,
            "available": len(expiry_summaries) > 0,
            "current_price": current_price,
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "pc_volume_ratio": round(total_put_vol / total_call_vol, 3) if total_call_vol else None,
            "pc_oi_ratio": round(total_put_oi / total_call_oi, 3) if total_call_oi else None,
            "atm_iv_call": atm_iv_call,
            "atm_iv_put": atm_iv_put,
            "iv_skew": iv_skew,
            "expirations": expiry_summaries,
        }

        cache.set(key, result)
        record_vintage("options")
        return result

    except Exception as e:
        logger.warning("Options flow failed for %s: %s", ticker, e)
        record_vintage("options")
        return unavailable
