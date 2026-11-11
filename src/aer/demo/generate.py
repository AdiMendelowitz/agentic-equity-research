"""Deterministic synthetic price series with a known, injectable conditional-mean signal.

The signal is a moving sum of past standard-normal shocks: mu[t] depends only on shocks eps[t-1..t-LOOKBACK] and never
on realised returns, so it cannot feed back on itself. Nothing here touches the network or the wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from aer.ml.constants import PREDICTION_HORIZON_DAYS
from aer.utils.trading_calendar import add_trading_days

DEFAULT_END: date = date(2026, 10, 30)
SIGMA_LOW: float = 0.010
SIGMA_HIGH: float = 0.020
STAY_PROBABILITY: float = 0.98
LOOKBACK: int = 10
_SEED_STRIDE: int = 1000


@dataclass(frozen=True)
class SyntheticSeries:
    ticker: str
    prices: pd.DataFrame
    mu: pd.Series
    sigma: pd.Series
    signal: float
    seed: int
    expected_forward: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _sessions(n_sessions: int, end: date) -> pd.DatetimeIndex:
    last = add_trading_days(end, 0)
    days = [add_trading_days(last, -k) for k in range(n_sessions - 1, -1, -1)]
    return pd.DatetimeIndex(pd.to_datetime(days))


def _sigma_path(n: int, rng: np.random.Generator) -> np.ndarray:
    draws = rng.random(n)
    state = 0
    out = np.empty(n)
    for t in range(n):
        if draws[t] > STAY_PROBABILITY:
            state = 1 - state
        out[t] = SIGMA_HIGH if state else SIGMA_LOW
    return out


def _mu(eps: np.ndarray, signal: float, sigma_ref: float) -> np.ndarray:
    n = len(eps)
    mu = np.zeros(n)
    if n > LOOKBACK:
        cumulative = np.concatenate(([0.0], np.cumsum(eps)))
        window_sum = cumulative[LOOKBACK:n] - cumulative[0 : n - LOOKBACK]
        mu[LOOKBACK:] = signal * sigma_ref * window_sum / np.sqrt(LOOKBACK)
    return mu


def _expected_forward(eps: np.ndarray, signal: float, sigma_ref: float) -> np.ndarray:
    n = len(eps)
    out = np.zeros(n)
    if n < LOOKBACK:
        return out
    for i in range(LOOKBACK):
        weight = min(PREDICTION_HORIZON_DAYS, LOOKBACK - i)
        out[LOOKBACK - 1 :] += weight * eps[LOOKBACK - 1 - i : n - i]
    return out * signal * sigma_ref / np.sqrt(LOOKBACK)


def synthetic_series(
    ticker: str,
    n_sessions: int = 700,
    seed: int = 0,
    signal: float = 0.0,
    end: date = DEFAULT_END,
    constant_sigma: float | None = None,
) -> SyntheticSeries:
    rng = np.random.default_rng(seed)
    index = _sessions(n_sessions, end)
    if constant_sigma is None:
        sigma = _sigma_path(n_sessions, rng)
        sigma_ref = (SIGMA_LOW + SIGMA_HIGH) / 2
    else:
        sigma = np.full(n_sessions, constant_sigma)
        sigma_ref = constant_sigma
    eps = rng.standard_normal(n_sessions)
    gap_z, high_z, low_z, vol_z = (rng.standard_normal(n_sessions) for _ in range(4))
    mu = _mu(eps, signal, sigma_ref)
    r = mu + sigma * eps
    close = 100.0 * np.exp(np.cumsum(r))
    prev_close = np.concatenate(([100.0], close[:-1]))
    open_ = prev_close * np.exp(0.25 * sigma * gap_z)
    high = np.maximum(open_, close) * np.exp(0.5 * sigma * np.abs(high_z))
    low = np.minimum(open_, close) * np.exp(-0.5 * sigma * np.abs(low_z))
    prices = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.round(1_000_000.0 * np.exp(0.3 * vol_z)),
            "Stock Splits": np.zeros(n_sessions),
        },
        index=index,
    )
    return SyntheticSeries(
        ticker=ticker,
        prices=prices,
        mu=pd.Series(mu, index=index, name="mu"),
        sigma=pd.Series(sigma, index=index, name="sigma"),
        signal=signal,
        seed=seed,
        expected_forward=pd.Series(_expected_forward(eps, signal, sigma_ref), index=index, name="expected_forward"),
    )


def synthetic_universe(
    n_tickers: int = 5,
    n_sessions: int = 700,
    seed: int = 0,
    signal: float = 0.0,
    constant_sigma: float | None = None,
) -> list[SyntheticSeries]:
    return [
        synthetic_series(f"SYN{i + 1}", n_sessions, seed * _SEED_STRIDE + i, signal, constant_sigma=constant_sigma)
        for i in range(n_tickers)
    ]
