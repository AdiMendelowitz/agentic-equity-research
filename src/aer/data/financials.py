"""Financial statements and derived ratios. Primary: SEC EDGAR (edgartools). Fallback: yfinance."""

import logging
import threading
import time
from typing import Any

import pandas as pd
import yfinance as yf

from aer.config import get_settings
from aer.utils.cache import data_cache

logger = logging.getLogger(__name__)

# EDGAR rate limiter — SEC enforces 10 req/s; target 1 req/s to be conservative
# since financials are cached 24h and concurrency is low.
_edgar_lock = threading.Lock()
_edgar_last_call: float = 0.0
_EDGAR_MIN_INTERVAL: float = 1.0  # seconds between EDGAR requests


def _edgar_wait() -> None:
    global _edgar_last_call
    with _edgar_lock:
        elapsed = time.time() - _edgar_last_call
        wait = _EDGAR_MIN_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)
        _edgar_last_call = time.time()


def _safe_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        if isinstance(val, (pd.Series, pd.DataFrame, list)):
            return None
        scalar = val.item() if hasattr(val, "item") else val
        return float(scalar) if not pd.isna(scalar) else None
    except Exception:
        return None


def _pct_diff(a: float, b: float) -> float:
    """Signed % difference: (a/b - 1)*100. Returns 0 if b is 0."""
    return (a / b - 1) * 100 if b else 0.0


def _stmt_to_records(df: pd.DataFrame, fields: list[str], max_periods: int = 4) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    records = []
    for col in df.columns[:max_periods]:
        label = col.date().isoformat() if hasattr(col, "date") else str(col)
        row: dict[str, Any] = {"period": label}
        for field in fields:
            row[field] = _safe_float(df[col].get(field))
        records.append(row)
    return records


def _get_yf_statements(ticker: str) -> dict[str, Any]:
    """Fetch financial statements from yfinance."""
    tk = yf.Ticker(ticker)

    income_records = _stmt_to_records(
        tk.income_stmt,
        ["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "EBITDA", "Basic EPS"],
    )
    balance_records = _stmt_to_records(
        tk.balance_sheet,
        [
            "Total Assets",
            "Total Liabilities Net Minority Interest",
            "Long Term Debt",
            "Cash And Cash Equivalents",
            "Stockholders Equity",
        ],
    )
    cashflow_records = _stmt_to_records(
        tk.cash_flow,
        ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"],
    )

    for cf in cashflow_records:
        if cf.get("Free Cash Flow") is None:
            ocf = cf.get("Operating Cash Flow")
            capex = cf.get("Capital Expenditure")
            if ocf is not None and capex is not None:
                cf["Free Cash Flow"] = ocf - abs(capex)

    return {
        "ticker": ticker,
        "income": income_records,
        "balance": balance_records,
        "cashflow": cashflow_records,
    }


def _get_edgar_statements(ticker: str) -> dict[str, Any] | None:
    """Fetch the latest 10-K from SEC EDGAR via edgartools. Returns None on any failure."""
    try:
        from edgar import Company, set_identity  # type: ignore

        identity = get_settings().edgar_identity
        if not identity or "example.com" in identity or "name@example.com" in identity:
            logger.debug(
                "EDGAR_IDENTITY not configured — set 'App Name name@example.com' in .env. "
                "Skipping EDGAR; using yfinance only."
            )
            return None
        set_identity(identity)

        _edgar_wait()
        for attempt in range(3):
            try:
                company = Company(ticker)
                filing = company.get_filings(form="10-K").latest(1)
                break
            except Exception as exc:
                exc_str = str(exc).lower()
                if ("429" in exc_str or "rate" in exc_str or "throttl" in exc_str) and attempt < 2:
                    backoff = 2.0 ** (attempt + 1)
                    logger.warning("EDGAR rate limited for %s — retrying in %.0fs", ticker, backoff)
                    time.sleep(backoff)
                else:
                    raise
        if filing is None:
            return None

        # edgartools >= 3.x: .financials is on the parsed report object, not EntityFiling
        report = filing.obj()
        if report is None:
            return None

        financials = report.financials
        metrics = financials.get_financial_metrics()

        period = "edgar-latest"
        income_record: dict[str, Any] = {
            "period": period,
            "Total Revenue": _safe_float(metrics.get("revenue")),
            "Gross Profit": None,
            "Operating Income": _safe_float(metrics.get("operating_income")),
            "Net Income": _safe_float(metrics.get("net_income")),
            "EBITDA": None,
            "Basic EPS": None,
        }
        balance_record: dict[str, Any] = {
            "period": period,
            "Total Assets": _safe_float(metrics.get("total_assets")),
            "Total Liabilities Net Minority Interest": _safe_float(metrics.get("total_liabilities")),
            "Long Term Debt": None,
            "Cash And Cash Equivalents": None,
            "Stockholders Equity": _safe_float(metrics.get("stockholders_equity")),
        }
        cashflow_record: dict[str, Any] = {
            "period": period,
            "Operating Cash Flow": _safe_float(metrics.get("operating_cash_flow")),
            "Capital Expenditure": _safe_float(metrics.get("capital_expenditures")),
            "Free Cash Flow": _safe_float(metrics.get("free_cash_flow")),
        }
        ocf = cashflow_record.get("Operating Cash Flow")
        capex = cashflow_record.get("Capital Expenditure")
        if cashflow_record["Free Cash Flow"] is None and ocf is not None and capex is not None:
            cashflow_record["Free Cash Flow"] = ocf - abs(capex)

        return {
            "ticker": ticker,
            "income": [income_record],
            "balance": [balance_record],
            "cashflow": [cashflow_record],
        }

    except Exception as e:
        logger.warning("EDGAR statements failed for %s: %s", ticker, e)
        return None


def get_financial_statements(ticker: str) -> dict[str, Any]:
    """Annual income statement, balance sheet, and cash flow for the last 4 years."""
    ticker = ticker.upper()
    key = f"financials:statements:{ticker}"
    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    try:
        edgar_data = _get_edgar_statements(ticker)
        yf_data = _get_yf_statements(ticker)

        discrepancies: list[str] = []

        if edgar_data is not None:
            edgar_income = edgar_data["income"][0] if edgar_data["income"] else {}
            yf_income = yf_data["income"][0] if yf_data["income"] else {}

            pairs = [
                ("Revenue", "Total Revenue", 1e9),
                ("Net Income", "Net Income", 1e9),
                ("EPS", "Basic EPS", 1.0),
            ]
            for label, field, scale in pairs:
                e_val = edgar_income.get(field)
                y_val = yf_income.get(field)
                if e_val is not None and y_val is not None and y_val != 0:
                    diff = _pct_diff(e_val, y_val)
                    if abs(diff) > 5:
                        if scale >= 1e9:
                            discrepancies.append(
                                f"{label}: edgar=${e_val/scale:.1f}b vs yfinance=${y_val/scale:.1f}b ({diff:+.1f}%)"
                            )
                        else:
                            discrepancies.append(f"{label}: edgar=${e_val:.2f} vs yfinance=${y_val:.2f} ({diff:+.1f}%)")

            income_combined = edgar_data["income"] + [r for r in yf_data["income"] if r["period"] != "edgar-latest"]
            result: dict[str, Any] = {
                "ticker": ticker,
                "income": income_combined,
                "balance": edgar_data["balance"] + yf_data["balance"],
                "cashflow": edgar_data["cashflow"] + yf_data["cashflow"],
                "_sources": {"primary": "edgar"},
                "_discrepancies": discrepancies,
            }
        else:
            result = {
                **yf_data,
                "_sources": {"primary": "yfinance"},
                "_discrepancies": [],
            }

        cache.set(key, result)
        return result

    except Exception as e:
        logger.warning("Financial statements failed for %s: %s", ticker, e)
        return {"ticker": ticker, "_error": str(e)}


def get_key_ratios(ticker: str) -> dict[str, Any]:
    """Derived ratios: revenue growth, margins, FCF, debt/equity, net cash."""
    ticker = ticker.upper()
    key = f"financials:ratios:{ticker}"
    cache = data_cache()

    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    try:
        stmts = get_financial_statements(ticker)
        income = stmts["income"]
        balance = stmts["balance"]
        cashflow = stmts["cashflow"]

        ratios: dict[str, Any] = {"ticker": ticker}

        yf_income = [r for r in income if r.get("period") != "edgar-latest"]
        if len(yf_income) >= 2:
            rev0 = yf_income[0].get("Total Revenue")
            rev1 = yf_income[1].get("Total Revenue")
            if rev0 is not None and rev1 is not None and rev1 != 0:
                ratios["revenue_growth_yoy"] = round((rev0 / rev1 - 1) * 100, 2)

        if income:
            rev = income[0].get("Total Revenue")
            gp = income[0].get("Gross Profit")
            oi = income[0].get("Operating Income")
            ni = income[0].get("Net Income")
            if rev is not None and rev != 0 and gp is not None:
                ratios["gross_margin"] = round(gp / rev * 100, 2)
            if rev is not None and rev != 0 and oi is not None:
                ratios["operating_margin"] = round(oi / rev * 100, 2)
            if rev is not None and rev != 0 and ni is not None:
                ratios["net_margin"] = round(ni / rev * 100, 2)

        if cashflow and balance:
            fcf = cashflow[0].get("Free Cash Flow")
            equity = balance[0].get("Stockholders Equity")
            debt = balance[0].get("Long Term Debt") or 0.0
            cash = balance[0].get("Cash And Cash Equivalents") or 0.0
            assets = balance[0].get("Total Assets")

            if balance[0].get("Long Term Debt") is None:
                logger.warning("Long Term Debt missing for %s — using 0 for ratio calculations", ticker)
            if fcf is not None and equity:
                ratios["fcf_on_equity"] = round(fcf / equity * 100, 2)
            if equity:
                ratios["debt_to_equity"] = round(debt / equity, 3)
            if assets:
                ratios["debt_to_assets"] = round(debt / assets, 3)
            ratios["net_cash"] = cash - debt

        ratios["revenue_history"] = [
            {"period": r["period"], "value": r.get("Total Revenue")}
            for r in income
            if r.get("Total Revenue") is not None
        ]
        ratios["fcf_history"] = [
            {"period": r["period"], "value": r.get("Free Cash Flow")}
            for r in cashflow
            if r.get("Free Cash Flow") is not None
        ]
        ratios["eps_history"] = [
            {"period": r["period"], "value": r.get("Basic EPS")} for r in income if r.get("Basic EPS") is not None
        ]

        cache.set(key, ratios)
        return ratios

    except Exception as e:
        logger.warning("Key ratios failed for %s: %s", ticker, e)
        return {"ticker": ticker, "_error": str(e)}
