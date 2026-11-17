"""Vectorised daily backtest engine: one instrument, one signal series, explicit costs.

Timing convention
 - The signal observed at the close of day t sets the position held from the close of day t to the close of day
   t+1, so position[t] = signal[t-1] and the first position is 0. The last signal is therefore never traded.
 - Gross return of day t = position[t] * (price[t] / price[t-1] - 1); the first day is 0.

Cost convention
 - cost_bps is a round trip (entry plus exit). Each unit of absolute position change costs cost_bps / 2 / 10,000
   and is charged on the day the new position starts. A flip from +1 to -1 is two units.
 - Net return = gross return minus cost.

Look-ahead safety
 - Net return on day t depends only on prices up to and including day t and on signals up to and including day
   t-1, so changing any later price leaves earlier net returns unchanged.

Dependencies: pandas, numpy and empyrical only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import empyrical
import numpy as np
import pandas as pd

from aer.ml.constants import ROUND_TRIP_COST_BPS

Mode = Literal["long_only", "long_short"]

_MODES: tuple[str, ...] = ("long_only", "long_short")
_ALLOWED_SIGNALS: frozenset[float] = frozenset({-1.0, 0.0, 1.0})
_BPS_PER_UNIT: float = 10_000.0


@dataclass(frozen=True, eq=False)
class BacktestResult:
    """Outcome of one backtest run.

    Attributes:
        net_returns: Daily return after costs, indexed like the input prices; the first day is 0.
        gross_returns: Daily return before costs.
        positions: Position held on each day (-1, 0 or 1); the first day is 0.
        turnover: Sum of absolute position changes over the whole run (a flip counts as 2).
        n_trades: Number of days on which the position changed.
        metrics: annual_return, sharpe_ratio and max_drawdown (empyrical, daily periodicity, computed on net
            returns) and hit_rate (share of days with a non-zero position whose net return is above 0; None when
            there is no such day). The empyrical figures can be NaN when the series is too short or has no
            dispersion.
    """

    net_returns: pd.Series
    gross_returns: pd.Series
    positions: pd.Series
    turnover: float
    n_trades: int
    metrics: dict[str, float | None]


def _check_series(name: str, value: object) -> pd.Series:
    """Return the value when it is a numeric pandas Series, otherwise raise."""
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(value).__name__}")
    if not pd.api.types.is_numeric_dtype(value.dtype):
        raise ValueError(f"{name} must be numeric, got dtype {value.dtype}")
    return value


def _validate(prices: pd.Series, signals: pd.Series, mode: str, cost_bps: float) -> None:
    """Raise ValueError (TypeError for a non-Series argument) when an input breaks the engine contract."""
    prices = _check_series("prices", prices)
    signals = _check_series("signals", signals)
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {list(_MODES)}, got {mode!r}")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError(f"cost_bps must be a finite number >= 0, got {cost_bps}")
    if len(prices) < 2:
        raise ValueError(f"at least 2 rows are required, got {len(prices)}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError(f"prices must be indexed by a DatetimeIndex, got {type(prices.index).__name__}")
    if not prices.index.equals(signals.index):
        raise ValueError("signals must have exactly the same index as prices")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("the index must be monotonic increasing")
    if not prices.index.is_unique:
        raise ValueError("the index must not contain duplicate timestamps")
    if prices.isna().any():
        raise ValueError("prices contain NaN")
    if signals.isna().any():
        raise ValueError("signals contain NaN")
    if not np.isfinite(prices.to_numpy(dtype="float64")).all() or (prices <= 0).any():
        raise ValueError("prices must be finite and strictly positive")
    bad = sorted(set(signals.astype("float64").unique()) - _ALLOWED_SIGNALS)
    if bad:
        raise ValueError(f"signals must take values in {{-1, 0, 1}}, found {bad}")


def _hit_rate(positions: pd.Series, net_returns: pd.Series) -> float | None:
    """Share of days with a non-zero position whose net return is above 0; None when no such day exists."""
    active = positions != 0
    if not active.any():
        return None
    return float((net_returns[active] > 0).mean())


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    *,
    mode: Mode = "long_only",
    cost_bps: float = ROUND_TRIP_COST_BPS,
) -> BacktestResult:
    """Backtest one signal series against one price series.

    Args:
        prices: Daily closes on a monotonic increasing DatetimeIndex, strictly positive, without NaN.
        signals: Signal observed at each close, on the same index, with values in {-1, 0, 1}.
        mode: "long_only" maps a signal of -1 to 0; "long_short" keeps it as a short position.
        cost_bps: Round-trip cost in basis points; half of it is charged per unit of absolute position change.

    Returns:
        BacktestResult with net and gross daily returns, positions, turnover, trade count and metrics.

    Raises:
        ValueError: On a mismatched or non-monotonic index, NaN, non-positive prices, signal values outside
            {-1, 0, 1}, fewer than 2 rows, an unknown mode or an invalid cost.
        TypeError: When prices or signals is not a pandas Series.
    """
    _validate(prices, signals, mode, cost_bps)

    wanted = signals.astype("float64")
    if mode == "long_only":
        wanted = wanted.clip(lower=0.0)
    positions = wanted.shift(1, fill_value=0.0).rename("position")

    price_change = (prices.astype("float64") / prices.astype("float64").shift(1) - 1.0).fillna(0.0)
    gross = (positions * price_change).rename("gross_return")

    # position[0] is always 0, so the first difference is 0 by construction
    units_traded = positions.diff().abs().fillna(0.0)
    cost = units_traded * (cost_bps / 2.0 / _BPS_PER_UNIT)
    net = (gross - cost).rename("net_return")

    metrics: dict[str, float | None] = {
        "annual_return": float(empyrical.annual_return(net, period="daily")),
        "sharpe_ratio": float(empyrical.sharpe_ratio(net, risk_free=0.0, period="daily")),
        "max_drawdown": float(empyrical.max_drawdown(net)),
        "hit_rate": _hit_rate(positions, net),
    }
    return BacktestResult(
        net_returns=net,
        gross_returns=gross,
        positions=positions,
        turnover=float(units_traded.sum()),
        n_trades=int((units_traded > 0).sum()),
        metrics=metrics,
    )
