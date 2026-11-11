"""AAPL 2020 4:1 split adjustment tests.

yfinance (verified 1.3.0) always applies stock-split adjustments, even with
auto_adjust=False - so neither price path shows a raw ~4x discontinuity. This gate
therefore tests three real truths against a committed offline fixture:

  - Close_true_unadjusted (reconstructed from the documented split factor) shows the
      genuine ~4x split discontinuity - the real contemporaneous traded price.
  - Close_raw (auto_adjust=False) is split-CONTINUOUS - proving splits are always
      applied - and differs from Close_adjusted only by the dividend-adjustment factor,
      which is the look-ahead get_price_history_raw actually avoids.

The fixture is the enforcing offline gate; the live test is supplementary (non-gating).
"""

from pathlib import Path

import pandas as pd
import pytest

FIXTURE = Path(__file__).parent.parent / "fixtures" / "aapl_2020_split.csv"
SPLIT_DATE = pd.Timestamp("2020-08-31")


def _load() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, index_col=0, parse_dates=True)


def _pre_post(df: pd.DataFrame, column: str) -> tuple[float, float]:
    pre = float(df[df.index < SPLIT_DATE][column].iloc[-1])
    post = float(df[df.index >= SPLIT_DATE][column].iloc[0])
    return pre, post


class TestData03SplitFixture:
    def test_true_unadjusted_shows_4x_discontinuity(self) -> None:
        """The genuine traded price (reconstructed from the split factor) drops
        ~4x at the split - the real corporate-action discontinuity."""
        pre, post = _pre_post(_load(), "Close_true_unadjusted")
        ratio = pre / post
        assert 3.5 < ratio < 4.5, f"Expected ~4x true-unadjusted ratio, got {ratio:.2f}"

    def test_raw_is_split_continuous(self) -> None:
        """yfinance auto_adjust=False is split-ADJUSTED: Close_raw is continuous
        across the split (no 4x jump), proving splits are always applied."""
        pre, post = _pre_post(_load(), "Close_raw")
        ratio = pre / post
        assert 0.8 < ratio < 1.2, f"Close_raw should be split-continuous, got ratio {ratio:.2f}"

    def test_adjusted_is_continuous(self) -> None:
        """Close_adjusted (auto_adjust=True) is also continuous across the split."""
        pre, post = _pre_post(_load(), "Close_adjusted")
        ratio = pre / post
        assert 0.8 < ratio < 1.2, f"Close_adjusted discontinuity at split: ratio {ratio:.2f}"

    def test_adjusted_differs_from_raw_by_dividends(self) -> None:
        """The adjusted and raw paths diverge by the dividend-adjustment factor
        (a few percent), NOT by the 4x split - this is the look-ahead the raw path avoids."""
        df = _load()
        adj = float(df.loc[SPLIT_DATE, "Close_adjusted"])
        raw = float(df.loc[SPLIT_DATE, "Close_raw"])
        diff_frac = abs(adj - raw) / raw
        assert (
            0.005 < diff_frac < 0.15
        ), f"adjusted vs raw should differ by the dividend factor (few %), got {diff_frac:.3%}"


@pytest.mark.live
def test_raw_continuous_and_differs_from_adjusted_live() -> None:
    """Supplementary (non-gating): live yfinance confirms raw is split-continuous and
    adjusted != raw across the 2020 split. Skips (not fails) when data is unavailable."""
    import yfinance as yf

    adj = yf.download("AAPL", start="2020-08-24", end="2020-09-05", auto_adjust=True, progress=False)
    raw = yf.download("AAPL", start="2020-08-24", end="2020-09-05", auto_adjust=False, progress=False)
    if adj.empty or raw.empty:
        pytest.skip("yfinance unavailable")

    def _close(df: pd.DataFrame) -> pd.Series:
        c = df["Close"]
        return c.iloc[:, 0] if getattr(c, "ndim", 1) > 1 else c

    raw_close = _close(raw)
    raw_close.index = pd.to_datetime(raw_close.index).tz_localize(None)
    pre = float(raw_close[raw_close.index < SPLIT_DATE].iloc[-1])
    post = float(raw_close[raw_close.index >= SPLIT_DATE].iloc[0])
    assert 0.5 < pre / post < 1.5, "live raw Close should be split-continuous (no 4x jump)"

    adj_close = _close(adj)
    adj_close.index = pd.to_datetime(adj_close.index).tz_localize(None)
    adj_split = float(adj_close[adj_close.index >= SPLIT_DATE].iloc[0])
    assert abs(adj_split - post) > 0.0, "adjusted and raw should differ (dividend factor)"
