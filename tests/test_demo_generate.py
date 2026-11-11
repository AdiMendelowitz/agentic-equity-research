from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal, assert_series_equal

from aer.demo.generate import DEFAULT_END, LOOKBACK, SIGMA_HIGH, SIGMA_LOW, synthetic_series, synthetic_universe
from aer.ml.constants import PREDICTION_HORIZON_DAYS
from aer.ml.features import build_feature_matrix
from aer.utils.trading_calendar import add_trading_days

LONG = 5000
SIGMA = 0.015


def test_same_arguments_give_identical_output() -> None:
    a = synthetic_series("SYN1", 300, seed=3, signal=0.4)
    b = synthetic_series("SYN1", 300, seed=3, signal=0.4)
    assert_frame_equal(a.prices, b.prices)
    assert_series_equal(a.mu, b.mu)
    assert_series_equal(a.sigma, b.sigma)
    assert_series_equal(a.expected_forward, b.expected_forward)


def test_different_seeds_differ() -> None:
    a = synthetic_series("SYN1", 100, seed=1)
    b = synthetic_series("SYN1", 100, seed=2)
    assert not a.prices["Close"].equals(b.prices["Close"])


def test_signal_zero_has_zero_mu_and_zero_expected_forward() -> None:
    series = synthetic_series("SYN1", 400, seed=5, signal=0.0)
    assert (series.mu == 0.0).all()
    assert (series.expected_forward == 0.0).all()


def test_signal_nonzero_has_nonzero_mu_after_the_lookback() -> None:
    series = synthetic_series("SYN1", 400, seed=5, signal=0.4)
    assert (series.mu.iloc[:LOOKBACK] == 0.0).all()
    assert (series.mu.iloc[LOOKBACK:] != 0.0).any()
    assert (series.expected_forward.iloc[: LOOKBACK - 1] == 0.0).all()


def test_constant_sigma_gives_a_constant_series_and_default_has_two_regimes() -> None:
    constant = synthetic_series("SYN1", 400, seed=2, signal=0.3, constant_sigma=SIGMA)
    assert (constant.sigma == SIGMA).all()
    regimes = synthetic_series("SYN1", 2000, seed=2, signal=0.3)
    assert set(regimes.sigma.unique()) == {SIGMA_LOW, SIGMA_HIGH}


def test_ohlc_consistency_and_columns() -> None:
    p = synthetic_series("SYN1", 500, seed=7, signal=0.15).prices
    assert list(p.columns) == ["Open", "High", "Low", "Close", "Volume", "Stock Splits"]
    assert (p["Low"] <= p[["Open", "Close"]].min(axis=1)).all()
    assert (p["High"] >= p[["Open", "Close"]].max(axis=1)).all()
    assert (p["Volume"] > 0).all()
    assert (p["Stock Splits"] == 0).all()
    assert p["Close"].pct_change().abs().max() < 0.2


def test_index_is_unique_increasing_nyse_sessions() -> None:
    n = 700
    idx = synthetic_series("SYN1", n).prices.index
    assert idx.tz is None
    assert len(idx) == n
    assert idx.is_unique
    assert idx.is_monotonic_increasing
    assert idx[-1].date() <= DEFAULT_END
    assert all(add_trading_days(ts.date(), 0) == ts.date() for ts in idx)


def test_universe_tickers_distinct_seeds_and_constant_sigma_passthrough() -> None:
    universe = synthetic_universe(n_tickers=3, n_sessions=100, seed=4, constant_sigma=SIGMA)
    assert [s.ticker for s in universe] == ["SYN1", "SYN2", "SYN3"]
    assert len({s.seed for s in universe}) == 3
    assert all((s.sigma == SIGMA).all() for s in universe)
    assert not universe[0].prices["Close"].equals(universe[1].prices["Close"])


def test_standardised_shock_sum_is_approximately_standard_normal() -> None:
    """m = mu / (signal * sigma_ref) is a sum of 10 shocks over sqrt(10). Over 5,000 overlapping windows the mean has
    a standard error of about 0.045 and the standard deviation of about 0.03, so the bounds 0.15 and 0.15 sit more
    than three standard errors out."""
    series = synthetic_series("SYN1", LONG, seed=11, signal=0.6, constant_sigma=SIGMA)
    m = series.mu.to_numpy()[LOOKBACK:] / (0.6 * SIGMA)
    assert abs(float(m.mean())) < 0.15
    assert abs(float(m.std()) - 1.0) < 0.15


def test_no_explosive_trend_at_the_strongest_signal() -> None:
    series = synthetic_series("SYN1", LONG, seed=11, signal=0.6, constant_sigma=SIGMA)
    daily = np.log(series.prices["Close"]).diff().dropna()
    assert abs(float(daily.mean())) < 0.2 * SIGMA


def test_expected_forward_tracks_the_realised_forward_return_at_the_analytic_correlation() -> None:
    """rho = sqrt(18 s^2 / (5 + 21 s^2 + 2 sqrt(10) s)) = 0.630 at s = 0.6. The sampling error over 5,000 sessions
    with overlapping windows is about 0.03, so the 0.15 tolerance is several standard errors wide."""
    s = 0.6
    series = synthetic_series("SYN1", LONG, seed=11, signal=s, constant_sigma=SIGMA)
    log_close = np.log(series.prices["Close"].to_numpy())
    h = PREDICTION_HORIZON_DAYS
    forward = log_close[h:] - log_close[:-h]
    expected = series.expected_forward.to_numpy()[: LONG - h]
    rho = float(np.corrcoef(expected[LOOKBACK:], forward[LOOKBACK:])[0, 1])
    analytic = float(np.sqrt(18 * s**2 / (5 + 21 * s**2 + 2 * np.sqrt(10) * s)))
    assert rho > 0
    assert abs(rho - analytic) < 0.15


def test_build_feature_matrix_accepts_prices_without_network() -> None:
    series = synthetic_series("SYN1", 700, seed=0, signal=0.4, constant_sigma=SIGMA)
    matrix = build_feature_matrix("SYN1", prices=series.prices)
    assert isinstance(matrix, pd.DataFrame)
    assert len(matrix) > 0
    assert "label" in matrix.columns


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(seed=st.integers(min_value=0, max_value=50), signal=st.sampled_from([0.0, 0.15, 0.4]))
def test_ohlc_bounds_hold_for_any_seed_and_signal(seed: int, signal: float) -> None:
    p = synthetic_series("SYN1", 80, seed=seed, signal=signal).prices
    assert (p["Low"] <= p[["Open", "Close"]].min(axis=1)).all()
    assert (p["High"] >= p[["Open", "Close"]].max(axis=1)).all()
