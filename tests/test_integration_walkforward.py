"""Walk-forward validation: one live-shaped run and hermetic checks of the history cutoff, fold layout and failures."""

from datetime import date
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import aer.risk.walkforward as wf
import aer.simulation.signal_generator as signal_generator
from aer.risk import WalkForwardResult, run_walk_forward
from aer.utils.wfo_constants import TOTAL_MIN_BARS, WFO_N_FOLDS, WFO_OOS_SIZE_BARS


def _prices(n_bars: int, start: str = "2020-01-02") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=n_bars)
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n_bars)))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000_000.0},
        index=index,
    )


@pytest.fixture
def hermetic(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the price fetch, the signal generator and the factor fetch; record what the generator was given."""
    seen: dict[str, Any] = {"prices": _prices(TOTAL_MIN_BARS), "max_is_date": [], "fail": False}

    def fake_generate(history: pd.DataFrame, signal_date: pd.Timestamp, tickers: list[str]) -> list[SimpleNamespace]:
        if seen["fail"]:
            raise ValueError("synthetic signal failure")
        seen["max_is_date"].append((history.index.max(), signal_date))
        return [SimpleNamespace(action="BUY")]

    monkeypatch.setattr(wf, "get_price_history_raw", lambda ticker: seen["prices"])
    monkeypatch.setattr(wf, "get_factor_loadings", lambda ticker, as_of_date: None)
    monkeypatch.setattr(signal_generator, "generate", fake_generate)
    return seen


@pytest.mark.live
class TestWalkForwardIntegration:
    """Live walk-forward run: fetches real prices, so it is excluded from CI by the live marker."""

    def test_run_walk_forward_aapl(self) -> None:
        """run_walk_forward(AAPL) on real prices returns five well-formed folds that end on or before as_of_date."""
        ticker = "AAPL"
        as_of_date = date.today()

        wfr = run_walk_forward(ticker, as_of_date)

        assert isinstance(wfr, WalkForwardResult)
        assert wfr.ticker == ticker
        assert wfr.as_of_date == as_of_date
        assert not wfr.insufficient_history, "AAPL has more than 577 bars; insufficient means the fetch or cut failed"
        assert len(wfr.folds) == WFO_N_FOLDS
        assert 0 <= wfr.n_folds_completed <= WFO_N_FOLDS
        for fold in wfr.folds:
            assert fold.n_is_bars >= 252
            assert fold.n_oos_bars == WFO_OOS_SIZE_BARS
            assert fold.oos_end <= as_of_date
        if wfr.mean_oos_drawdown is not None:
            assert wfr.mean_oos_drawdown <= 0


class TestWalkForwardHermetic:
    """No network: the fetch, the signal generator and the factor fetch are patched."""

    def test_short_history_is_insufficient(self, hermetic: dict[str, Any]) -> None:
        hermetic["prices"] = _prices(TOTAL_MIN_BARS - 1)
        wfr = run_walk_forward("SYN", date(2030, 1, 1))
        assert wfr.insufficient_history is True
        assert wfr.folds == []
        assert wfr.n_folds_completed == 0

    def test_bars_after_as_of_date_do_not_count_towards_history(self, hermetic: dict[str, Any]) -> None:
        prices = _prices(TOTAL_MIN_BARS + 200)
        hermetic["prices"] = prices
        as_of = prices.index[TOTAL_MIN_BARS - 2].date()
        wfr = run_walk_forward("SYN", as_of)
        assert wfr.insufficient_history is True
        assert wfr.folds == []

    def test_folds_end_at_the_last_bar_on_or_before_as_of_date(self, hermetic: dict[str, Any]) -> None:
        prices = _prices(TOTAL_MIN_BARS + 200)
        hermetic["prices"] = prices
        cutoff = TOTAL_MIN_BARS + 100
        as_of = prices.index[cutoff - 1].date()
        wfr = run_walk_forward("SYN", as_of)
        assert wfr.insufficient_history is False
        assert wfr.n_folds_completed == WFO_N_FOLDS
        assert wfr.folds[-1].oos_end == as_of
        assert all(f.oos_end <= as_of and f.is_end <= as_of for f in wfr.folds)
        assert wfr.folds[0].n_is_bars == 252 + 100

    def test_signals_never_see_bars_after_the_in_sample_window(self, hermetic: dict[str, Any]) -> None:
        hermetic["prices"] = _prices(TOTAL_MIN_BARS + 50)
        run_walk_forward("SYN", date(2030, 1, 1))
        assert hermetic["max_is_date"]
        assert all(last_is_bar < signal_date for last_is_bar, signal_date in hermetic["max_is_date"])

    def test_signal_failure_fails_the_fold(self, hermetic: dict[str, Any]) -> None:
        hermetic["fail"] = True
        wfr = run_walk_forward("SYN", date(2030, 1, 1))
        assert wfr.insufficient_history is False
        assert wfr.n_folds_completed == 0
        assert all(f.fold_error is not None and "signal generation failed" in f.fold_error for f in wfr.folds)
        assert all(f.oos_sharpe_long is None and f.oos_sharpe_ls is None for f in wfr.folds)
        assert wfr.mean_oos_sharpe_long is None
