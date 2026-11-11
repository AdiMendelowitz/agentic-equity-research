"""
This module never fetches data. It only processes the history DataFrame passed by
the caller, which must contain only rows with index < signal_date.
"""

import dataclasses
import logging
from typing import Literal

import pandas as pd

from aer.agents.technical import _classify_signals
from aer.indicators.technical import compute_indicators
from aer.utils.python_replacements import signal_action_from_classification

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Signal:
    """Trading signal for one ticker on one day."""

    ticker: str
    action: Literal["BUY", "HOLD", "SELL"]
    confidence: int  # 0–100
    reasoning: str
    signal_date: pd.Timestamp


def _format_ticker_indicators(ticker: str, history: pd.DataFrame) -> str:
    """Return a one-line indicator summary for one ticker."""
    try:
        ticker_df = history.xs(ticker, level=1, axis=1).copy()
    except KeyError:
        return f"  {ticker}: data_unavailable"

    if len(ticker_df) < 30:
        return f"  {ticker}: insufficient_history ({len(ticker_df)} rows)"

    ind = compute_indicators(ticker_df)

    def fmt(v: float | None, d: int = 2) -> str:
        """Format a float to d decimal places, or 'N/A' if None."""
        return f"{v:.{d}f}" if v is not None else "N/A"

    bb_pct_b: float | None = None
    bu, bl = ind.get("bb_upper"), ind.get("bb_lower")
    if bu is not None and bl is not None and (bu - bl) != 0:
        bb_pct_b = (ind["price"] - bl) / (bu - bl)

    return (
        f"  {ticker}: price={fmt(ind['price'])}"
        f" RSI={fmt(ind['rsi'], 1)}"
        f" MACD(line={fmt(ind['macd'], 3)},sig={fmt(ind['macd_signal'], 3)},hist={fmt(ind['macd_hist'], 3)})"
        f" BB%B={fmt(bb_pct_b)}"
    )


def generate(
    history: pd.DataFrame,
    signal_date: pd.Timestamp,
    tickers: list[str],
) -> list[Signal]:
    """
    # LLM required: N/A - replaced with deterministic Python classifier

    Generate BUY/HOLD/SELL signals for all tickers using only pre-signal-date data.

    Args:
        history: DataFrame with MultiIndex columns (field, ticker). Must contain ONLY
                 rows with index < signal_date - caller is responsible for slicing.
        signal_date: The trading day for which signals are generated.
        tickers: Tickers to generate signals for.

    Returns:
        list[Signal] - one per ticker. Missing data defaults to HOLD.

    Raises:
        ValueError: if history contains any row at or after signal_date.
    """
    if not history.empty:

        def _naive(ts: pd.Timestamp) -> pd.Timestamp:
            return ts.tz_localize(None) if ts.tzinfo is not None else ts

        max_idx = _naive(history.index.max())
        sd_naive = _naive(signal_date)
        if max_idx >= sd_naive:
            raise ValueError(
                f"Temporal integrity violation: history.index.max()={history.index.max()} "
                f">= signal_date={signal_date}. Caller must pass df.loc[df.index < signal_date]."
            )

    results: list[Signal] = []
    for ticker in tickers:
        try:
            ticker_df = history.xs(ticker, level=1, axis=1).copy()
        except KeyError:
            results.append(
                Signal(
                    ticker=ticker,
                    action="HOLD",
                    confidence=0,
                    reasoning="No history data available",
                    signal_date=signal_date,
                )
            )
            continue

        if len(ticker_df) < 30:
            results.append(
                Signal(
                    ticker=ticker,
                    action="HOLD",
                    confidence=0,
                    reasoning=f"Insufficient history ({len(ticker_df)} rows)",
                    signal_date=signal_date,
                )
            )
            continue

        ind = compute_indicators(ticker_df)
        signals = _classify_signals(ind)
        overall = signals.get("overall", "NEUTRAL")
        action = signal_action_from_classification(overall)
        # Confidence from indicator agreement
        signal_values = [v for k, v in signals.items() if k != "overall"]
        agreed = sum(1 for v in signal_values if v == overall)
        confidence = int(agreed / max(len(signal_values), 1) * 100)

        results.append(
            Signal(
                ticker=ticker,
                action=action,
                confidence=confidence,
                reasoning=f"Technical signals: {overall} ({agreed}/{len(signal_values)} indicators agree)",
                signal_date=signal_date,
            )
        )

    return results
