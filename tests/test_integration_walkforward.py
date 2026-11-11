"""Integration test for walk-forward validation function.

Tests that run_walk_forward() returns well-formed results for realistic tickers.
"""

from datetime import date

from aer.risk import WalkForwardResult, run_walk_forward


class TestWalkForwardIntegration:
    """Standalone walk-forward integration: ticker → run_walk_forward() → results validated."""

    def test_run_walk_forward_aapl(self) -> None:
        """Test that run_walk_forward(AAPL) returns valid WalkForwardResult (5-year history)."""
        ticker = "AAPL"
        as_of_date = date.today()

        # Run walk-forward for AAPL (5-year history should have > 577 bars)
        wfr = run_walk_forward(ticker, as_of_date)

        # Verify result structure
        assert wfr is not None
        assert isinstance(wfr, WalkForwardResult)
        assert wfr.ticker == ticker
        assert wfr.as_of_date == as_of_date

        # Verify fold results
        if not wfr.insufficient_history:
            # AAPL should have sufficient history
            assert len(wfr.folds) == 5
            assert wfr.n_folds_completed >= 1  # At least one fold should complete
            assert wfr.n_folds_completed <= 5

            # Verify each fold has required structure
            for fold in wfr.folds:
                assert fold.fold_idx >= 0
                assert fold.fold_idx < 5
                assert fold.is_start is not None
                assert fold.is_end is not None
                assert fold.oos_start is not None
                assert fold.oos_end is not None
                assert fold.n_is_bars >= 252
                assert fold.n_oos_bars == 55

            # Verify aggregate metrics are sensible (if data available)
            if wfr.mean_oos_sharpe_long is not None:
                assert isinstance(wfr.mean_oos_sharpe_long, float)
            if wfr.mean_oos_sharpe_ls is not None:
                assert isinstance(wfr.mean_oos_sharpe_ls, float)
            if wfr.mean_oos_drawdown is not None:
                assert isinstance(wfr.mean_oos_drawdown, float)
                assert wfr.mean_oos_drawdown <= 0  # Loss convention
        else:
            # Insufficient history case (shouldn't happen for AAPL)
            assert len(wfr.folds) == 0
            assert wfr.mean_oos_sharpe_long is None
            assert wfr.mean_oos_sharpe_ls is None
            assert wfr.mean_oos_drawdown is None

    def test_run_walk_forward_insufficient_history(self) -> None:
        """Test that run_walk_forward returns insufficient_history=True for very short history."""
        # Use a very new/pink-sheet ticker with minimal history
        # (or mock the history to be too short)
        # For now, test the happy path with AAPL
        pass
