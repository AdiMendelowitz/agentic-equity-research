"""a price frame must not depend on whether it came from the network or the cache.

Before the fix the cold path returned yfinance's frame as-is (index possibly tz-aware, NaN
floats) while the warm path rebuilt it from the cached payload (tz-naive, via _cacheable_to_df),
so date comparisons downstream behaved differently on the first and second call of a day
(pelosi_agent.py carried an inline tz_localize(None) patch for exactly this). The cold path now
returns the same round-trip the warm path reads, so cold == warm by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import aer.data.prices as prices


class _DictCache:
    def __init__(self):
        self.d = {}

    def get(self, k):
        return self.d.get(k)

    def set(self, k, v, *a, **kw):
        self.d[k] = v


def _aware_frame():
    idx = pd.date_range("2026-07-06", periods=4, freq="B", tz="America/New_York", name="Date")
    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, np.nan, 4.0],
            "High": [1, 2, 3, 4.0],
            "Low": [1, 2, 3, 4.0],
            "Close": [1.5, 2.5, 3.5, 4.5],
            "Volume": [10, 20, 30, 40],
        },
        index=idx,
    )


@pytest.mark.parametrize("fn", ["get_price_history", "get_price_history_raw"])
def test_cold_and_warm_frames_are_identical(monkeypatch, fn):
    cache = _DictCache()
    monkeypatch.setattr(prices, "data_cache", lambda: cache)
    monkeypatch.setattr(prices.yf, "download", lambda *a, **k: _aware_frame())
    monkeypatch.setattr(prices, "record_vintage", lambda *a, **k: None, raising=False)
    cold = getattr(prices, fn)("SPY")
    warm = getattr(prices, fn)("SPY")
    assert cold.index.tz is None
    pd.testing.assert_frame_equal(cold, warm)


@pytest.mark.parametrize("fn", ["get_price_history", "get_price_history_raw"])
def test_aware_index_spanning_dst_parses_and_matches(monkeypatch, fn):
    cache = _DictCache()
    idx = pd.date_range("2026-10-28", "2026-11-06", freq="B", tz="America/New_York", name="Date")  # DST ends 11-01
    frame = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": range(len(idx)), "Volume": 1}, index=idx)
    monkeypatch.setattr(prices, "data_cache", lambda: cache)
    monkeypatch.setattr(prices.yf, "download", lambda *a, **k: frame)
    monkeypatch.setattr(prices, "record_vintage", lambda *a, **k: None, raising=False)
    cold = getattr(prices, fn)("SPY")
    warm = getattr(prices, fn)("SPY")
    assert cold.index.tz is None and len(cold) == len(idx)
    pd.testing.assert_frame_equal(cold, warm)
