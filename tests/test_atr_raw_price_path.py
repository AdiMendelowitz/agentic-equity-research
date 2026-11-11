"""(HIGH, ruled 2026-09-23): ATR, support and the entry/stop prices derived from them are
signal computations, so they read the dividend-UNADJUSTED path (prices.py:115-124 states the
rule: the adjusted path retroactively rescales history on every dividend). Before the fix both
price_targets and the technical agent read get_price_history (adjusted), so entry_price and
stop_loss_price sat below the tradable price by the cumulative dividend factor."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd


def _frame(scale: float, n: int = 260) -> pd.DataFrame:
    idx = pd.bdate_range("2025-06-02", periods=n)
    base = np.linspace(100, 120, n) * scale
    return pd.DataFrame(
        {"Open": base, "High": base * 1.01, "Low": base * 0.99, "Close": base, "Volume": np.full(n, 1_000_000)},
        index=idx,
    )


RAW, ADJ = _frame(1.0), _frame(0.9)  # adjusted history sits ~10% lower, as after dividends


def test_price_targets_read_the_raw_path():
    from aer.data import price_targets as pt

    with (
        patch.object(pt, "get_raw_history_last_year", return_value=RAW, create=True),
        patch.object(pt, "get_price_history", return_value=ADJ, create=True),
    ):
        out = pt.compute_price_targets("ACME")
    # raw history ends at 120 (adjusted at 108): every level must be on the tradable raw scale
    assert out["entry_price"] > 110 and out["stop_loss_price"] > 105 and out["target_price"] > 115


def test_technical_agent_reads_the_raw_path():
    import aer.agents.technical as tech

    seen = {}

    def raw(ticker):
        seen["raw"] = ticker
        return RAW

    with (
        patch.object(tech, "get_raw_history_last_year", side_effect=raw, create=True),
        patch.object(tech, "get_price_history", side_effect=AssertionError("adjusted path used"), create=True),
        patch.object(tech, "agent_cache") as ac,
        patch.object(tech, "complete", side_effect=RuntimeError("no LLM in this test")) as llm,
    ):
        ac.return_value.get.return_value = None
        try:
            tech.run("ACME")
        except AssertionError:
            raise
        except Exception:
            pass  # downstream formatting is irrelevant here; only the price path is asserted
    # the LLM boundary is patched, so no provider (OpenRouter, NIM, Ollama, Anthropic) is called
    assert llm.call_count <= 1
    assert seen.get("raw") == "ACME"


def test_raw_last_year_slices_the_prefetch_warmed_3y_key(monkeypatch):
    import aer.data.prices as prices

    three = _frame(1.0, n=780)
    calls = []
    monkeypatch.setattr(prices, "get_price_history_raw", lambda t, *a, **k: calls.append((t, a, k)) or three)
    out = prices.get_raw_history_last_year("ACME")
    assert calls == [("ACME", (), {})]  # default period -> the 3y key the prefetch warms
    assert out.index[0] > three.index[-1] - pd.DateOffset(years=1) and out.index[-1] == three.index[-1]
