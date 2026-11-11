"""Institutional and insider ownership, short interest. Cross-validated with FinViz."""

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from aer.utils.cache import data_cache
from aer.utils.vintage_tracker import record_vintage

logger = logging.getLogger(__name__)


def _parse_pct(raw: Any) -> float | None:
    """Parse '0.57%' or '5.7%' → 0.0057. Returns None on failure."""
    try:
        s = str(raw).strip().rstrip("%")
        return float(s) / 100
    except Exception:
        return None


def _get_finviz_ownership(ticker: str) -> dict[str, Any]:
    """FinViz is unavailable: returns an empty dict."""
    logger.debug("FinViz ownership unavailable for %s", ticker)
    return {}


def get_ownership(ticker: str) -> dict[str, Any]:
    """Institutional holders, insider transactions, short interest. Cross-validated with FinViz."""
    ticker = ticker.upper()
    key = f"ownership:{ticker}"
    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        record_vintage("ownership")
        return cached  # type: ignore[return-value]

    result: dict[str, Any] = {"ticker": ticker}

    try:
        tk = yf.Ticker(ticker)

        try:
            info = tk.info or {}
            result["insider_pct"] = info.get("heldPercentInsiders")
            result["institution_pct"] = info.get("heldPercentInstitutions")
            result["short_float_pct"] = info.get("shortPercentOfFloat")
            result["short_ratio"] = info.get("shortRatio")
        except Exception as e:
            logger.warning("Ownership info failed for %s: %s", ticker, e)

        try:
            ih = tk.institutional_holders
            if ih is not None and not ih.empty:
                result["institutional_holders"] = [
                    {
                        "holder": str(row.get("Holder", "")),
                        "shares": int(row.get("Shares")) if pd.notna(row.get("Shares")) else None,
                        "pct_held": float(row.get("% Out")) if pd.notna(row.get("% Out")) else None,
                    }
                    for _, row in ih.head(10).iterrows()
                ]
        except Exception as e:
            logger.warning("Institutional holders failed for %s: %s", ticker, e)

        try:
            it = tk.insider_transactions
            if it is not None and not it.empty:
                transactions: list[dict[str, Any]] = []
                for _, row in it.head(15).iterrows():
                    transactions.append(
                        {
                            "insider": str(row.get("Insider", "")),
                            "title": str(row.get("Position", "")),
                            "transaction": str(row.get("Transaction", "")),
                            "shares": int(row.get("Shares")) if pd.notna(row.get("Shares")) else None,
                            "value": float(row.get("Value")) if pd.notna(row.get("Value")) else None,
                            "date": str(row.get("Start Date", "")),
                        }
                    )
                result["insider_transactions"] = transactions

                buys = sum(
                    1
                    for t in transactions
                    if any(word in t["transaction"].lower() for word in ("buy", "purchase", "acquisition"))
                )
                sells = sum(
                    1
                    for t in transactions
                    if any(word in t["transaction"].lower() for word in ("sell", "sale", "disposition"))
                )
                result["insider_buys_recent"] = buys
                result["insider_sells_recent"] = sells
                # 999.0 sentinel = all buys, no sells (strongest insider buy signal); float("inf") is not JSON-serializable
                result["insider_buy_sell_ratio"] = round(buys / sells, 2) if sells else (999.0 if buys > 0 else None)
        except Exception as e:
            logger.warning("Insider transactions failed for %s: %s", ticker, e)

        finviz = _get_finviz_ownership(ticker)
        discrepancies: list[str] = []
        yf_short = result.get("short_float_pct")
        fvz_short = finviz.get("short_float_pct")

        if yf_short is not None and fvz_short is not None:
            if yf_short != 0 and abs((fvz_short / yf_short - 1) * 100) > 10:
                discrepancies.append(f"Short Float: yf={yf_short*100:.1f}% vs finviz={fvz_short*100:.1f}%")
            short_source = "both"
        elif yf_short is None and fvz_short is not None:
            result["short_float_pct"] = fvz_short
            short_source = "finviz"
        else:
            short_source = "yfinance" if yf_short is not None else "none"

        result["_sources"] = {"short_float": short_source}
        result["_discrepancies"] = discrepancies
        result["_finviz"] = finviz

        cache.set(key, result)
        record_vintage("ownership")
        return result

    except Exception as e:
        logger.warning("Ownership fetch failed for %s: %s", ticker, e)
        record_vintage("ownership")
        return result
