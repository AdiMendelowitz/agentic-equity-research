"""Fama-French 5-factor OLS regression: get_factor_loadings implementation.

Module-level imports of get_factor_data and get_price_history_raw allow
monkeypatch.setattr("aer.risk.factors.get_factor_data", ...) to work correctly
in unit tests without deferred-import indirection.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from aer.data.factors import get_factor_data
from aer.data.prices import get_price_history_raw
from aer.risk.models import FactorLoadings
from aer.utils.constants import FACTOR_MIN_MONTHS, FACTOR_PREFERRED_MONTHS, FACTOR_ROLLING_WINDOW_MONTHS

logger = logging.getLogger(__name__)

try:
    import statsmodels.api as sm

    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

_FACTOR_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
_PARAM_MAP = {"const": "alpha", "Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RMW": "rmw", "CMA": "cma"}


def get_factor_loadings(ticker: str, as_of_date: date) -> FactorLoadings:
    """Compute Fama-French 5-factor loadings for *ticker* as of *as_of_date*.

    Single FF5 OLS regression on 60-month rolling window of monthly returns.
    FF3 betas (mkt_rf, smb, hml) are the first three coefficients of the single regression.
    Alpha is annualised: (1 + monthly_intercept)^12 - 1.
    t/p stats use HAC (Newey-West, 3 lags) standard errors.

    Args:
        ticker: Equity ticker symbol.
        as_of_date: Point-in-time cutoff; no factor or price data after this date.

    Returns:
        FactorLoadings. Returns insufficient_history=True when:
        - statsmodels not available
        - aligned monthly observations < FACTOR_MIN_MONTHS (36)
        - any OLS error
    """
    if not _HAS_STATSMODELS:
        logger.warning("statsmodels unavailable — get_factor_loadings returning insufficient_history=True")
        return FactorLoadings(insufficient_history=True)

    try:
        # Load factor data and enforce as_of_date look-ahead boundary
        factors = get_factor_data(as_of=as_of_date)
        factors.index = pd.to_datetime(factors.index)  # normalise date/DatetimeIndex → DatetimeIndex
        factors = factors[factors.index <= pd.Timestamp(as_of_date)]

        # Load price history — request 5+ years for 60-month rolling window
        prices_raw = get_price_history_raw(ticker, period="5y")
        if isinstance(prices_raw, pd.DataFrame):
            if prices_raw.empty or "Close" not in prices_raw.columns:
                return FactorLoadings(insufficient_history=True)
            close_series = prices_raw["Close"]
        elif isinstance(prices_raw, pd.Series):
            if prices_raw.empty:
                return FactorLoadings(insufficient_history=True)
            close_series = prices_raw
        else:
            return FactorLoadings(insufficient_history=True)

        # Monthly price returns
        monthly_prices = close_series.resample("ME").last()
        monthly_returns = monthly_prices.pct_change().dropna()

        # Monthly factor returns via geometric compounding (sum is arithmetically incorrect).
        # Use .prod() directly to avoid pandas 2.x apply() MultiIndex return-value changes.
        monthly_factors = (1 + factors).resample("ME").prod() - 1

        # Inner join on month-end date, then enforce as_of_date on aligned index
        aligned = monthly_returns.to_frame("ret").join(monthly_factors, how="inner")
        aligned = aligned[aligned.index <= pd.Timestamp(as_of_date)]
        window = aligned.tail(FACTOR_ROLLING_WINDOW_MONTHS)

        n_obs = len(window)

        if n_obs < FACTOR_MIN_MONTHS:
            return FactorLoadings(insufficient_history=True, n_obs=n_obs)

        low_data_quality = n_obs < FACTOR_PREFERRED_MONTHS

        # OLS: excess return = alpha + Σ(beta_i * factor_i) + ε
        excess = window["ret"] - window["RF"]
        X = sm.add_constant(window[_FACTOR_COLS], has_constant="add")
        fit = sm.OLS(excess, X).fit()

        # HAC (Newey-West, 3 lags) for t-statistics and p-values.
        # get_robustcov_results returns numpy arrays for tvalues/pvalues in some statsmodels
        # builds, so we re-wrap with the OLS params index to allow string-keyed access.
        fit_hac = fit.get_robustcov_results(cov_type="HAC", maxlags=3)
        param_index = fit.params.index
        tvalues_s = pd.Series(fit_hac.tvalues, index=param_index)
        pvalues_s = pd.Series(fit_hac.pvalues, index=param_index)

        # Alpha annualised via geometric compounding; betas are raw monthly OLS coefficients
        monthly_alpha = float(fit.params["const"])
        alpha = round((1 + monthly_alpha) ** 12 - 1, 6)

        betas = {_PARAM_MAP[k]: round(float(fit.params[k]), 6) for k in _FACTOR_COLS}

        all_keys = ["const"] + _FACTOR_COLS
        t_stats = {_PARAM_MAP[k]: round(float(tvalues_s[k]), 4) for k in all_keys}
        p_values = {_PARAM_MAP[k]: round(float(pvalues_s[k]), 4) for k in all_keys}

        return FactorLoadings(
            alpha=alpha,
            mkt_rf=betas["mkt_rf"],
            smb=betas["smb"],
            hml=betas["hml"],
            rmw=betas["rmw"],
            cma=betas["cma"],
            r_squared=round(float(fit.rsquared), 6),
            t_stats=t_stats,
            p_values=p_values,
            n_obs=n_obs,
            insufficient_history=False,
            low_data_quality=low_data_quality,
        )

    except Exception:
        logger.warning(
            "get_factor_loadings(%s, %s) — returning insufficient_history=True", ticker, as_of_date, exc_info=True
        )
        return FactorLoadings(insufficient_history=True)
