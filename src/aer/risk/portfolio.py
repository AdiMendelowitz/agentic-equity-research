"""Portfolio risk metrics: VaR, CVaR, downside deviation, drawdown detail."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from aer.risk.models import RiskMetrics

try:
    import empyrical as ep

    _HAS_EMPYRICAL = True
except ImportError:
    _HAS_EMPYRICAL = False

logger = logging.getLogger(__name__)

MIN_RISK_POINTS: int = 30  # minimum for any risk computation
MIN_VAR_95_POINTS: int = 60  # minimum for statistically defensible VaR/CVaR at 95%
MIN_VAR_99_POINTS: int = 250  # minimum for statistically defensible VaR/CVaR at 99%
ANNUALIZATION_FACTOR: float = math.sqrt(252)


def compute_drawdown_detail(equity: pd.Series) -> dict:
    """Vectorized max drawdown interval: peak date, trough date, recovery/duration days.

    Returns dict with keys: dd_start (date|None), dd_end (date|None),
    recovery_days (int|None, trough->recovery), dd_duration_days (int|None, peak->recovery).
    """
    if len(equity) < 2:
        return {"dd_start": None, "dd_end": None, "recovery_days": None, "dd_duration_days": None}

    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max  # negative or zero

    max_dd = float(drawdowns.min())
    if max_dd >= 0:
        return {"dd_start": None, "dd_end": None, "recovery_days": None, "dd_duration_days": None}

    # Trough: index with maximum relative drawdown
    trough_idx = drawdowns.idxmin()

    # Peak preceding the trough: last time running_max equals peak_val before trough
    peak_val = float(running_max.loc[trough_idx])
    pre_trough = equity.loc[:trough_idx]
    peak_idx = pre_trough[pre_trough >= peak_val].index[-1]

    # Recovery: first date post-trough where equity >= peak_val (trough excluded — it cannot be a recovery point)
    trough_iloc = equity.index.get_loc(trough_idx)
    post_trough = equity.iloc[trough_iloc + 1 :]
    recovery_candidates = post_trough[post_trough >= peak_val]

    dd_start = peak_idx.date() if hasattr(peak_idx, "date") else peak_idx
    dd_end = trough_idx.date() if hasattr(trough_idx, "date") else trough_idx

    if not recovery_candidates.empty:
        recovery_idx = recovery_candidates.index[0]
        recovery_date = recovery_idx.date() if hasattr(recovery_idx, "date") else recovery_idx
        recovery_days = (recovery_date - dd_end).days  # trough -> recovery (standard)
        dd_duration_days = (recovery_date - dd_start).days  # peak -> recovery
    else:
        recovery_days = None
        dd_duration_days = None

    return {
        "dd_start": dd_start,
        "dd_end": dd_end,
        "recovery_days": recovery_days,
        "dd_duration_days": dd_duration_days,
    }


def compute_risk(returns: pd.Series) -> RiskMetrics:
    """Compute portfolio risk metrics from a daily returns series.

    Args:
        returns: pd.Series of daily decimal returns (negative = loss).
                 Should have a DatetimeIndex for correct drawdown dates.

    Returns:
        RiskMetrics with all fields populated or None based on sample size guards.
    """
    # C7 fix: Sort + dedup check on the DatetimeIndex before any computation
    returns = returns.sort_index()
    if returns.index.duplicated().any():
        logger.warning("compute_risk: duplicate dates in returns index — deduplicating")
        returns = returns[~returns.index.duplicated(keep="last")]
    returns = returns.dropna()

    # INSUFFICIENT HISTORY guard
    n = len(returns)
    if n < MIN_RISK_POINTS:
        return RiskMetrics(insufficient_history=True)

    # VaR/CVaR TIER FLAGS
    insufficient_var95 = n < MIN_VAR_95_POINTS
    insufficient_var99 = n < MIN_VAR_99_POINTS

    # DOWNSIDE DEVIATION (MAR=0): denominator is N_total (not N_neg) per Sortino/Price 1994
    neg_mask = returns < 0
    neg_returns = returns[neg_mask]
    if len(neg_returns) > 0:
        downside_dev = round(
            ANNUALIZATION_FACTOR * math.sqrt(float((neg_returns**2).sum()) / n),
            6,
        )
    else:
        downside_dev = None  # no negative returns -> not computable

    # VaR/CVaR HISTORICAL (empyrical or numpy fallback)
    var_95_hist: float | None = None
    cvar_95_hist: float | None = None
    var_99_hist: float | None = None
    cvar_99_hist: float | None = None

    if not insufficient_var95:
        if _HAS_EMPYRICAL:
            var_95_hist = min(round(float(ep.value_at_risk(returns, cutoff=0.05)), 6), 0.0)
            cvar_95_hist = min(round(float(ep.conditional_value_at_risk(returns, cutoff=0.05)), 6), 0.0)
        else:
            _raw_pct5 = float(np.percentile(returns, 5))
            var_95_hist = min(round(_raw_pct5, 6), 0.0)
            tail_95 = returns[returns <= _raw_pct5]
            cvar_95_hist = min(round(float(tail_95.mean()), 6), 0.0) if len(tail_95) > 0 else None

        if not insufficient_var99:
            if _HAS_EMPYRICAL:
                var_99_hist = min(round(float(ep.value_at_risk(returns, cutoff=0.01)), 6), 0.0)
                cvar_99_hist = min(round(float(ep.conditional_value_at_risk(returns, cutoff=0.01)), 6), 0.0)
            else:
                _raw_pct1 = float(np.percentile(returns, 1))
                var_99_hist = min(round(_raw_pct1, 6), 0.0)
                tail_99 = returns[returns <= _raw_pct1]
                cvar_99_hist = min(round(float(tail_99.mean()), 6), 0.0) if len(tail_99) > 0 else None

    # VaR/CVaR PARAMETRIC (scipy.stats.norm)
    var_95_param: float | None = None
    cvar_95_param: float | None = None
    var_99_param: float | None = None
    cvar_99_param: float | None = None

    if not insufficient_var95:
        mu = float(returns.mean())
        sigma = float(returns.std())
        if sigma > 0:
            var_95_param = min(round(float(norm.ppf(0.05, loc=mu, scale=sigma)), 6), 0.0)
            cvar_95_param = min(round(float(mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05), 6), 0.0)
            if not insufficient_var99:
                var_99_param = min(round(float(norm.ppf(0.01, loc=mu, scale=sigma)), 6), 0.0)
                cvar_99_param = min(round(float(mu - sigma * norm.pdf(norm.ppf(0.01)) / 0.01), 6), 0.0)

    # DRAWDOWN DETAIL (vectorized, using equity curve reconstructed from returns)
    try:
        equity = (1 + returns).cumprod()
        dd_detail = compute_drawdown_detail(equity)
    except Exception as e:
        logger.warning("compute_risk: drawdown detail failed: %s", e)
        dd_detail = {"dd_start": None, "dd_end": None, "recovery_days": None, "dd_duration_days": None}

    return RiskMetrics(
        dd_start=dd_detail["dd_start"],
        dd_end=dd_detail["dd_end"],
        recovery_days=dd_detail["recovery_days"],
        dd_duration_days=dd_detail["dd_duration_days"],
        downside_dev=downside_dev,
        var_95_hist=var_95_hist,
        cvar_95_hist=cvar_95_hist,
        var_95_param=var_95_param,
        cvar_95_param=cvar_95_param,
        var_99_hist=var_99_hist,
        cvar_99_hist=cvar_99_hist,
        var_99_param=var_99_param,
        cvar_99_param=cvar_99_param,
        insufficient_history=False,
        insufficient_var95=insufficient_var95,
        insufficient_var99=insufficient_var99,
    )
