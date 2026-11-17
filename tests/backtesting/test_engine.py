"""Hand-checkable tests for the backtest engine.

Fixture prices: 100, 110, 99, 99, 108.9 on five consecutive business days.
Daily price changes: day 1: 110 / 100 - 1 = +0.10; day 2: 99 / 110 - 1 = -0.10; day 3: 99 / 99 - 1 = 0;
day 4: 108.9 / 99 - 1 = +0.10. Day 0 has no previous close, so its change is 0.
With the default cost of 10 basis points a round trip, one unit of position change costs 5 / 10,000 = 0.0005.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from aer.backtesting import BacktestResult, run_backtest
from aer.ml.constants import ROUND_TRIP_COST_BPS

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "src" / "aer" / "backtesting"
TOL = 1e-12


def _index(n: int = 5) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _prices() -> pd.Series:
    return pd.Series([100.0, 110.0, 99.0, 99.0, 108.9], index=_index())


def _signals(values: list[float]) -> pd.Series:
    return pd.Series(values, index=_index(len(values)))


# Signals observed at each close: [1, 1, 0, 1, 1].
# Positions (signal shifted by one day, first position 0): [0, 1, 1, 0, 1].
# Position changes: day 1 +1 (entry), day 2 0, day 3 -1 (exit), day 4 +1 (entry).
# Gross: [0, 1 * 0.10, 1 * -0.10, 0 * 0, 1 * 0.10] = [0, 0.10, -0.10, 0, 0.10].
# Cost: [0, 0.0005, 0, 0.0005, 0.0005].
# Net: [0, 0.0995, -0.10, -0.0005, 0.0995].
LONG_SIGNALS = [1.0, 1.0, 0.0, 1.0, 1.0]
LONG_NET = [0.0, 0.0995, -0.10, -0.0005, 0.0995]


def _long_result() -> BacktestResult:
    return run_backtest(_prices(), _signals(LONG_SIGNALS))


def test_position_is_signal_shifted_by_one_day() -> None:
    result = _long_result()
    assert result.positions.tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert result.positions.index.equals(_index())


def test_first_position_is_zero_even_when_the_first_signal_is_long() -> None:
    result = _long_result()
    assert result.positions.iloc[0] == 0.0
    assert result.gross_returns.iloc[0] == 0.0
    assert result.net_returns.iloc[0] == 0.0


def test_gross_returns_follow_the_shifted_position() -> None:
    result = _long_result()
    assert result.gross_returns.to_numpy() == pytest.approx([0.0, 0.10, -0.10, 0.0, 0.10], abs=TOL)


def test_net_returns_are_gross_minus_cost_on_entry_and_exit() -> None:
    # Entry on day 1 and day 4, exit on day 3: each costs 0.0005. Day 2 holds the position and costs nothing.
    result = _long_result()
    assert result.net_returns.to_numpy() == pytest.approx(LONG_NET, abs=TOL)
    assert (result.gross_returns - result.net_returns).to_numpy() == pytest.approx(
        [0.0, 0.0005, 0.0, 0.0005, 0.0005], abs=TOL
    )


def test_turnover_and_trade_count_for_long_only() -> None:
    # Changes are +1, 0, -1, +1 on days 1 to 4: turnover 3, three days with a change.
    result = _long_result()
    assert result.turnover == pytest.approx(3.0)
    assert result.n_trades == 3
    assert isinstance(result.n_trades, int)
    assert isinstance(result.turnover, float)


# Signals [-1, -1, 1, 0, -1] in long-short mode.
# Positions: [0, -1, -1, 1, 0]. Changes: day 1 -1 (short entry), day 2 0, day 3 +2 (flip from -1 to +1),
# day 4 -1 (exit). Turnover 1 + 2 + 1 = 4; n_trades 3.
# Gross: [0, -1 * 0.10, -1 * -0.10, 1 * 0, 0 * 0.10] = [0, -0.10, 0.10, 0, 0].
# Cost: [0, 0.0005, 0, 2 * 0.0005, 0.0005] = [0, 0.0005, 0, 0.001, 0.0005].
# Net: [0, -0.1005, 0.10, -0.001, -0.0005].
SHORT_SIGNALS = [-1.0, -1.0, 1.0, 0.0, -1.0]


def test_long_short_cost_on_entry_flip_and_exit() -> None:
    result = run_backtest(_prices(), _signals(SHORT_SIGNALS), mode="long_short")
    assert result.positions.tolist() == [0.0, -1.0, -1.0, 1.0, 0.0]
    assert result.gross_returns.to_numpy() == pytest.approx([0.0, -0.10, 0.10, 0.0, 0.0], abs=TOL)
    assert result.net_returns.to_numpy() == pytest.approx([0.0, -0.1005, 0.10, -0.001, -0.0005], abs=TOL)
    assert result.turnover == pytest.approx(4.0)
    assert result.n_trades == 3


def test_flip_costs_two_units() -> None:
    # Signals [1, -1, -1, -1, -1]: positions [0, 1, -1, -1, -1]; the flip on day 2 changes the position by 2.
    # Day 1 cost 0.0005, day 2 cost 2 * 0.0005 = 0.001; no further changes.
    result = run_backtest(_prices(), _signals([1.0, -1.0, -1.0, -1.0, -1.0]), mode="long_short")
    cost = result.gross_returns - result.net_returns
    assert cost.to_numpy() == pytest.approx([0.0, 0.0005, 0.001, 0.0, 0.0], abs=TOL)
    assert result.turnover == pytest.approx(3.0)
    assert result.n_trades == 2


def test_long_only_clips_short_signals_to_flat() -> None:
    # Same signals as the long-short case, clipped to [0, 0, 1, 0, 0].
    # Positions: [0, 0, 0, 1, 0]. Changes: day 3 +1, day 4 -1. Turnover 2; n_trades 2.
    # Gross: day 3 holds 1 * 0 = 0, so every gross figure is 0. Net: [0, 0, 0, -0.0005, -0.0005].
    result = run_backtest(_prices(), _signals(SHORT_SIGNALS), mode="long_only")
    assert result.positions.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert result.positions.min() >= 0.0
    assert result.gross_returns.to_numpy() == pytest.approx([0.0] * 5, abs=TOL)
    assert result.net_returns.to_numpy() == pytest.approx([0.0, 0.0, 0.0, -0.0005, -0.0005], abs=TOL)
    assert result.turnover == pytest.approx(2.0)
    assert result.n_trades == 2


def test_default_cost_is_the_shared_round_trip_constant() -> None:
    default = run_backtest(_prices(), _signals(LONG_SIGNALS))
    explicit = run_backtest(_prices(), _signals(LONG_SIGNALS), cost_bps=ROUND_TRIP_COST_BPS)
    assert_series_equal(default.net_returns, explicit.net_returns)


def test_cost_scales_with_cost_bps() -> None:
    # 20 basis points a round trip: 10 / 10,000 = 0.001 per unit. Net: [0, 0.099, -0.10, -0.001, 0.099].
    result = run_backtest(_prices(), _signals(LONG_SIGNALS), cost_bps=20.0)
    assert result.net_returns.to_numpy() == pytest.approx([0.0, 0.099, -0.10, -0.001, 0.099], abs=TOL)


def test_zero_cost_makes_net_equal_gross() -> None:
    result = run_backtest(_prices(), _signals(LONG_SIGNALS), cost_bps=0.0)
    assert_series_equal(result.net_returns, result.gross_returns, check_names=False)


# --- metrics ---


def test_hit_rate_counts_days_with_a_position_and_positive_net_return() -> None:
    # Net [0, 0.0995, -0.10, -0.0005, 0.0995] with positions [0, 1, 1, 0, 1]: held days 1, 2, 4.
    # Net above 0 on days 1 and 4 only: 2 / 3.
    result = _long_result()
    assert result.metrics["hit_rate"] == pytest.approx(2 / 3)


def test_hit_rate_long_short_example() -> None:
    # Net [0, -0.1005, 0.10, -0.001, -0.0005], positions [0, -1, -1, 1, 0]: held days 1, 2, 3.
    # Net above 0 on day 2 only: 1 / 3.
    result = run_backtest(_prices(), _signals(SHORT_SIGNALS), mode="long_short")
    assert result.metrics["hit_rate"] == pytest.approx(1 / 3)


def test_hit_rate_is_zero_when_every_held_day_loses() -> None:
    # Long-only clipped example: the only held day is day 3, net -0.0005, so 0 / 1.
    result = run_backtest(_prices(), _signals(SHORT_SIGNALS), mode="long_only")
    assert result.metrics["hit_rate"] == 0.0


def test_hit_rate_is_none_without_any_held_day() -> None:
    result = run_backtest(_prices(), _signals([0.0] * 5))
    assert result.metrics["hit_rate"] is None
    assert result.turnover == 0.0
    assert result.n_trades == 0
    assert result.net_returns.to_numpy() == pytest.approx([0.0] * 5, abs=TOL)


def test_hit_rate_is_none_when_only_the_last_signal_is_set() -> None:
    # The last signal is never traded because position[t] = signal[t-1].
    result = run_backtest(_prices(), _signals([0.0, 0.0, 0.0, 0.0, 1.0]))
    assert result.metrics["hit_rate"] is None
    assert result.n_trades == 0


def test_empyrical_metrics_match_explicit_numpy_formulae() -> None:
    # Read in the installed empyrical (stats.py): sharpe_ratio is nanmean(r - risk_free) / nanstd(..., ddof=1) *
    # sqrt(annualization_factor), with the daily factor 252 (periods.py); annual_return is
    # cum_returns_final(r, starting_value=1) ** (1 / (len(r) / 252)) - 1, over every row including day 0;
    # max_drawdown is the minimum of the drawdown series, which starts from a base of 100.
    result = _long_result()
    net = np.array(LONG_NET)
    n = len(net)

    expected_sharpe = net.mean() / net.std(ddof=1) * np.sqrt(252)
    assert result.metrics["sharpe_ratio"] == pytest.approx(expected_sharpe, rel=1e-9)

    # Growth factors 1, 1.0995, 0.9, 0.9995, 1.0995 multiply to the terminal value.
    terminal = np.prod(1.0 + net)
    expected_annual = terminal ** (252 / n) - 1.0
    assert result.metrics["annual_return"] == pytest.approx(expected_annual, rel=1e-9)

    # Wealth path 1, 1.0995, 0.98955, 0.98905..., 1.0874...; the peak is 1.0995 from day 1 onwards.
    # Deepest point is day 3: 0.9 * 0.9995 - 1 = -0.10045.
    wealth = np.cumprod(1.0 + net)
    expected_drawdown = (wealth / np.maximum.accumulate(wealth) - 1.0).min()
    assert expected_drawdown == pytest.approx(0.9 * 0.9995 - 1.0, abs=TOL)
    assert result.metrics["max_drawdown"] == pytest.approx(expected_drawdown, abs=1e-9)


def test_metrics_keys() -> None:
    assert set(_long_result().metrics) == {"annual_return", "sharpe_ratio", "max_drawdown", "hit_rate"}


def test_result_is_frozen() -> None:
    result = _long_result()
    with pytest.raises(AttributeError):
        result.turnover = 0.0  # type: ignore[misc]


# --- no look-ahead ---


@pytest.mark.parametrize("t", [1, 2, 3])
def test_changing_later_prices_leaves_earlier_net_returns_unchanged(t: int) -> None:
    base = run_backtest(_prices(), _signals(LONG_SIGNALS))
    mutated_prices = _prices()
    mutated_prices.iloc[t + 1 :] = mutated_prices.iloc[t + 1 :] * 3.7 + 11.0
    mutated = run_backtest(mutated_prices, _signals(LONG_SIGNALS))
    assert_series_equal(base.net_returns.iloc[: t + 1], mutated.net_returns.iloc[: t + 1])
    assert_series_equal(base.positions.iloc[: t + 1], mutated.positions.iloc[: t + 1])


@pytest.mark.parametrize("t", [1, 2, 3])
def test_changing_signals_from_day_t_leaves_net_returns_up_to_t_unchanged(t: int) -> None:
    # position[t] = signal[t-1], so a signal change on day t first acts on day t + 1.
    base = run_backtest(_prices(), _signals(LONG_SIGNALS), mode="long_short")
    flipped = _signals(LONG_SIGNALS)
    flipped.iloc[t:] = -1.0
    mutated = run_backtest(_prices(), flipped, mode="long_short")
    assert_series_equal(base.net_returns.iloc[: t + 1], mutated.net_returns.iloc[: t + 1])


# --- validation ---


def test_rejects_mismatched_index() -> None:
    other = pd.Series([1.0] * 5, index=pd.bdate_range("2024-02-01", periods=5))
    with pytest.raises(ValueError, match="same index"):
        run_backtest(_prices(), other)


def test_rejects_signals_of_different_length() -> None:
    with pytest.raises(ValueError, match="same index"):
        run_backtest(_prices(), _signals([1.0, 0.0, 1.0]))


def test_rejects_non_monotonic_index() -> None:
    prices = _prices().iloc[::-1]
    signals = _signals(LONG_SIGNALS).iloc[::-1]
    with pytest.raises(ValueError, match="monotonic increasing"):
        run_backtest(prices, signals)


def test_rejects_duplicate_timestamps() -> None:
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"])
    prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
    signals = pd.Series([0.0, 1.0, 1.0, 0.0], index=index)
    with pytest.raises(ValueError, match="duplicate"):
        run_backtest(prices, signals)


def test_rejects_non_datetime_index() -> None:
    prices = pd.Series([100.0, 101.0, 102.0])
    signals = pd.Series([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        run_backtest(prices, signals)


def test_rejects_nan_in_prices() -> None:
    prices = _prices()
    prices.iloc[2] = np.nan
    with pytest.raises(ValueError, match="prices contain NaN"):
        run_backtest(prices, _signals(LONG_SIGNALS))


def test_rejects_nan_in_signals() -> None:
    signals = _signals(LONG_SIGNALS)
    signals.iloc[1] = np.nan
    with pytest.raises(ValueError, match="signals contain NaN"):
        run_backtest(_prices(), signals)


@pytest.mark.parametrize("bad_price", [0.0, -5.0, np.inf])
def test_rejects_non_positive_or_infinite_prices(bad_price: float) -> None:
    prices = _prices()
    prices.iloc[3] = bad_price
    with pytest.raises(ValueError, match="strictly positive"):
        run_backtest(prices, _signals(LONG_SIGNALS))


@pytest.mark.parametrize("bad_value", [2.0, -2.0, 0.5])
def test_rejects_signal_values_outside_the_allowed_set(bad_value: float) -> None:
    signals = _signals(LONG_SIGNALS)
    signals.iloc[2] = bad_value
    with pytest.raises(ValueError, match="values in"):
        run_backtest(_prices(), signals)


def test_rejects_fewer_than_two_rows() -> None:
    prices = _prices().iloc[:1]
    signals = _signals(LONG_SIGNALS).iloc[:1]
    with pytest.raises(ValueError, match="at least 2 rows"):
        run_backtest(prices, signals)
    with pytest.raises(ValueError, match="at least 2 rows"):
        run_backtest(_prices().iloc[:0], _signals(LONG_SIGNALS).iloc[:0])


def test_two_rows_are_accepted() -> None:
    # Signals [1, 0] give positions [0, 1]; day 1 gross is 110 / 100 - 1 = 0.10, cost 0.0005, net 0.0995.
    result = run_backtest(_prices().iloc[:2], _signals(LONG_SIGNALS).iloc[:2])
    assert result.net_returns.to_numpy() == pytest.approx([0.0, 0.0995], abs=TOL)


def test_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        run_backtest(_prices(), _signals(LONG_SIGNALS), mode="short_only")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_cost", [-1.0, float("nan"), float("inf")])
def test_rejects_invalid_cost(bad_cost: float) -> None:
    with pytest.raises(ValueError, match="cost_bps"):
        run_backtest(_prices(), _signals(LONG_SIGNALS), cost_bps=bad_cost)


def test_rejects_non_series_inputs() -> None:
    with pytest.raises(TypeError, match="prices must be a pandas Series"):
        run_backtest([100.0, 101.0], _signals([0.0, 1.0]))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="signals must be a pandas Series"):
        run_backtest(_prices(), [0.0] * 5)  # type: ignore[arg-type]


def test_rejects_non_numeric_signals() -> None:
    signals = pd.Series(["long"] * 5, index=_index())
    with pytest.raises(ValueError, match="numeric"):
        run_backtest(_prices(), signals)


# --- package hygiene ---


def test_package_source_names_no_event_driven_framework() -> None:
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    assert files, f"no source files found under {PACKAGE_DIR}"
    pattern = re.compile(r"backtrader|cerebro|bt\.Strategy", re.IGNORECASE)
    offenders = [(path.name, match.group(0)) for path in files for match in pattern.finditer(path.read_text("utf-8"))]
    assert offenders == []
