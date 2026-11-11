"""Walk-forward out-of-sample validation: 5 expanding-window folds.

Implements run_walk_forward(ticker, as_of_date) which:
1. Fetches raw price history (split-adjusted, dividend-unadjusted)
2. Checks sufficient history (>= 577 bars)
3. Computes 5 expanding-window fold boundaries
4. Generates signals per fold (IS-only history)
5. Computes per-fold OOS Sharpe (long-only, long-short) and drawdown
6. Fetches Fama-French factors per fold (graceful None on unavailability)
7. Aggregates across folds (mean/std metrics, fold completion count)

Fold geometry
 - IS window: expanding from start (bar 0) to growing end
 - Purge gap: 10-bar non-trading period post-IS
 - OOS window: fixed 55-bar window post-purge-gap
 - 5 folds total; Fold 0 uses minimum 252-bar IS window
 - Total minimum bars: 577

Design:
 - Zero LLM calls: Python indicators only
 - MultiIndex adapter: wraps raw OHLCV for signal_generator.generate
 - Signal-price reconciliation both use raw prices from get_price_history_raw
 - Per-fold error tracking fold_error set on any exception; other folds complete
 - Per-fold factor data graceful None if Ken French absent; walk-forward continues
 - Seeding random.seed(42) at function entry for reproducibility
"""

from __future__ import annotations

import logging
import random
from datetime import date

import numpy as np
import pandas as pd

from aer.data.prices import get_price_history_raw
from aer.risk.factors import get_factor_loadings
from aer.risk.models import FoldResult, WalkForwardResult
from aer.utils.wfo_constants import (
    TOTAL_MIN_BARS,
    WFO_MIN_IS_BARS,
    WFO_N_FOLDS,
    WFO_OOS_SIZE_BARS,
    WFO_PURGE_GAP_BARS,
)

logger = logging.getLogger(__name__)

try:
    import empyrical as ep

    _HAS_EMPYRICAL = True
except ImportError:
    _HAS_EMPYRICAL = False


def _compute_fold_boundaries(num_bars: int) -> list[tuple[int, int, int, int]]:
    """Compute IS/OOS bar indices for 5 expanding-window folds.

    Args:
    num_bars: Total number of bars available in price history.

    Returns:
    List of (is_start_bar, is_end_bar, oos_start_bar, oos_end_bar) tuples.
    All indices are 0-based and inclusive of start, exclusive of end (Python slice semantics).
    """
    folds = []
    for fold_idx in range(WFO_N_FOLDS):
        # IS window: anchored at start (bar 0) to growing end
        is_start_bar = 0
        is_end_bar = WFO_MIN_IS_BARS + fold_idx * (WFO_PURGE_GAP_BARS + WFO_OOS_SIZE_BARS)

        # Purge gap: [is_end_bar, is_end_bar + WFO_PURGE_GAP_BARS)
        purge_end_bar = is_end_bar + WFO_PURGE_GAP_BARS

        # OOS window: [purge_end_bar, purge_end_bar + WFO_OOS_SIZE_BARS)
        oos_start_bar = purge_end_bar
        oos_end_bar = purge_end_bar + WFO_OOS_SIZE_BARS

        folds.append((is_start_bar, is_end_bar, oos_start_bar, oos_end_bar))

    return folds


def run_walk_forward(ticker: str, as_of_date: date) -> WalkForwardResult:
    """Run 5-fold walk-forward out-of-sample validation.

    Args:
        ticker: Equity ticker symbol.
        as_of_date: Snapshot date (passed to get_factor_loadings for time-consistency).

    Returns:
        WalkForwardResult with per-fold metrics, aggregates, and fold_error tracking.
        Returns insufficient_history=True if total bars < 577; all metrics are None.

    Raises:
        None — all exceptions caught per-fold; fold_error field set on failure.
    """
    # Seed for reproducibility
    random.seed(42)

    # Fetch raw price history raw prices matched to signal generation)
    try:
        price_history = get_price_history_raw(ticker)
    except Exception as e:
        logger.error(f"Failed to fetch price history for {ticker}: {e}")
        return WalkForwardResult(
            ticker=ticker,
            as_of_date=as_of_date,
            folds=[],
            insufficient_history=True,
        )

    num_bars = len(price_history)

    # Check sufficient history (hard cutoff at 577 bars; no partial walk-forward)
    if num_bars < TOTAL_MIN_BARS:
        logger.info(f"{ticker}: Insufficient history ({num_bars} bars < {TOTAL_MIN_BARS} required)")
        return WalkForwardResult(
            ticker=ticker,
            as_of_date=as_of_date,
            folds=[],
            insufficient_history=True,
        )

    # Compute fold boundaries
    fold_boundaries = _compute_fold_boundaries(num_bars)

    # Process each fold
    fold_results: list[FoldResult] = []
    n_folds_completed = 0
    n_folds_with_factor_alpha = 0

    for fold_idx, (is_start_bar, is_end_bar, oos_start_bar, oos_end_bar) in enumerate(fold_boundaries):
        fold_error: str | None = None
        oos_sharpe_long: float | None = None
        oos_sharpe_ls: float | None = None
        oos_drawdown: float | None = None
        factor_alpha: float | None = None
        factor_alpha_significant: bool = False

        try:
            # Slice IS history (IS-only, no OOS lookahead)
            is_history = price_history.iloc[is_start_bar:is_end_bar]
            is_start_date = is_history.index[0].date()
            is_end_date = is_history.index[-1].date()
            n_is_bars = len(is_history)

            # Slice OOS returns (for performance computation)
            oos_history = price_history.iloc[oos_start_bar:oos_end_bar]
            oos_start_date = oos_history.index[0].date()
            oos_end_date = oos_history.index[-1].date()
            n_oos_bars = len(oos_history)

            # Wrap IS history with MultiIndex for signal_generator.generate
            # signal_generator expects MultiIndex columns: (field, ticker)
            is_multiindex = pd.MultiIndex.from_tuples([(col, ticker) for col in is_history.columns])
            is_history_multiindex = is_history.copy()
            is_history_multiindex.columns = is_multiindex

            # Generate signals for all OOS bars one bar at a time, IS-only history)
            from aer.simulation.signal_generator import generate  # Local import to avoid circular dependency

            oos_signals: list[str] = []
            for oos_bar_idx in range(oos_start_bar, oos_end_bar):
                # Slice price history up to this OOS bar (but keep IS-only before is_end_bar)
                signal_date = price_history.index[oos_bar_idx]

                # Call signal_generator.generate() with IS-only history
                # (is_end_bar prevents OOS lookahead; is_history_multiindex contains [is_start:is_end] only)
                try:
                    signals = generate(is_history_multiindex, signal_date, [ticker])
                    if signals and len(signals) > 0:
                        oos_signals.append(signals[0].action)
                    else:
                        oos_signals.append("HOLD")
                except Exception as sig_err:
                    logger.warning(f"Signal generation failed for {ticker} at {signal_date}: {sig_err}")
                    oos_signals.append("HOLD")

            # Compute OOS returns (daily returns from closing prices)
            oos_close = oos_history["Close"]
            oos_returns = oos_close.pct_change().dropna()

            # Convert signal strings to numeric position sizes
            signal_numeric = np.array([1 if s == "BUY" else (-1 if s == "SELL" else 0) for s in oos_signals])

            # Trim signal_numeric to match oos_returns length (first return is NaN, dropped)
            signal_numeric = signal_numeric[1:]

            if len(oos_returns) == 0 or len(signal_numeric) == 0:
                raise ValueError(f"Empty OOS returns or signals for fold {fold_idx}")

            # Long-only position returns
            long_only_position = (signal_numeric == 1).astype(int)  # Only 1s become position
            long_only_returns = oos_returns.values * long_only_position

            # Long-short position returns
            ls_returns = oos_returns.values * signal_numeric

            # Compute Sharpe ratios (empyrical with risk_free=0)
            if _HAS_EMPYRICAL:
                try:
                    oos_sharpe_long = ep.sharpe_ratio(pd.Series(long_only_returns), risk_free=0.0, period="daily")
                    oos_sharpe_ls = ep.sharpe_ratio(pd.Series(ls_returns), risk_free=0.0, period="daily")
                except Exception as sharpe_err:
                    logger.warning(f"Sharpe calculation failed for fold {fold_idx}: {sharpe_err}")
                    oos_sharpe_long = None
                    oos_sharpe_ls = None

                # Compute max drawdown (empyrical)
                try:
                    oos_drawdown = ep.max_drawdown(pd.Series(long_only_returns))
                except Exception as dd_err:
                    logger.warning(f"Drawdown calculation failed for fold {fold_idx}: {dd_err}")
                    oos_drawdown = None
            else:
                logger.warning("empyrical not available; skipping Sharpe/drawdown computation")
                oos_sharpe_long = None
                oos_sharpe_ls = None
                oos_drawdown = None

            # Get Fama-French factor alpha per fold graceful None on unavailability)
            try:
                factor_loadings = get_factor_loadings(ticker, as_of_date=is_end_date)
                if factor_loadings and not factor_loadings.insufficient_history:
                    factor_alpha = factor_loadings.alpha
                    if factor_loadings.p_values and "alpha" in factor_loadings.p_values:
                        factor_alpha_significant = factor_loadings.p_values["alpha"] < 0.05
                    if factor_alpha is not None:
                        n_folds_with_factor_alpha += 1
            except Exception as factor_err:
                logger.debug(f"Factor data unavailable for {ticker} at {is_end_date}: {factor_err}")
                factor_alpha = None
                factor_alpha_significant = False

            n_folds_completed += 1

        except Exception as e:
            # Per-fold error handling
            fold_error = str(e)
            logger.warning(f"Fold {fold_idx} failed for {ticker}: {fold_error}")
            is_start_date = price_history.index[is_start_bar].date()
            is_end_date = price_history.index[is_end_bar - 1].date()
            oos_start_date = price_history.index[oos_start_bar].date()
            oos_end_date = price_history.index[oos_end_bar - 1].date()
            n_is_bars = is_end_bar - is_start_bar
            n_oos_bars = WFO_OOS_SIZE_BARS

        # Create FoldResult
        fold_result = FoldResult(
            fold_idx=fold_idx,
            is_start=is_start_date,
            is_end=is_end_date,
            n_is_bars=n_is_bars,
            oos_start=oos_start_date,
            oos_end=oos_end_date,
            n_oos_bars=n_oos_bars,
            oos_sharpe_long=oos_sharpe_long,
            oos_sharpe_ls=oos_sharpe_ls,
            oos_drawdown=oos_drawdown,
            factor_alpha=factor_alpha,
            factor_alpha_significant=factor_alpha_significant,
            fold_error=fold_error,
        )
        fold_results.append(fold_result)

    # Aggregate metrics across folds (skip None and NaN values)
    sharpe_long_values = np.array([f.oos_sharpe_long for f in fold_results if f.oos_sharpe_long is not None])
    sharpe_ls_values = np.array([f.oos_sharpe_ls for f in fold_results if f.oos_sharpe_ls is not None])
    drawdown_values = np.array([f.oos_drawdown for f in fold_results if f.oos_drawdown is not None])

    mean_oos_sharpe_long = float(np.nanmean(sharpe_long_values)) if len(sharpe_long_values) > 0 else None
    mean_oos_sharpe_ls = float(np.nanmean(sharpe_ls_values)) if len(sharpe_ls_values) > 0 else None
    std_oos_sharpe_long = float(np.nanstd(sharpe_long_values)) if len(sharpe_long_values) > 1 else None
    std_oos_sharpe_ls = float(np.nanstd(sharpe_ls_values)) if len(sharpe_ls_values) > 1 else None
    mean_oos_drawdown = float(np.nanmean(drawdown_values)) if len(drawdown_values) > 0 else None

    return WalkForwardResult(
        ticker=ticker,
        as_of_date=as_of_date,
        folds=fold_results,
        mean_oos_sharpe_long=mean_oos_sharpe_long,
        mean_oos_sharpe_ls=mean_oos_sharpe_ls,
        std_oos_sharpe_long=std_oos_sharpe_long,
        std_oos_sharpe_ls=std_oos_sharpe_ls,
        mean_oos_drawdown=mean_oos_drawdown,
        n_folds_completed=n_folds_completed,
        n_folds_with_factor_alpha=n_folds_with_factor_alpha,
        insufficient_history=False,
    )
