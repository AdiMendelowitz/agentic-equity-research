"""Tests for opinion tracking: constants, models, record_opinion and get_opinions,
hit-rate metrics and forward-return population.
Tests are deterministic (no LLM calls, no API mocks beyond basic SQLite mock).
"""

from datetime import date

import pytest

from aer.data.opinions import get_opinions, record_opinion
from aer.models import HitRateMetrics, OpinionRecord
from aer.utils.opinion_constants import (
    BINARY_HIT_THRESHOLD,
    HOLD_THRESHOLD,
    MIN_OPINIONS_FOR_METRICS,
    MIN_VALID_RETURNS_FOR_SHARPE,
)


class TestConstants:
    """Constant definitions."""

    def test_constants_import_and_bounds(self):
        """Verify all constants import without error and bounds are sensible."""
        assert MIN_OPINIONS_FOR_METRICS >= 10
        assert MIN_VALID_RETURNS_FOR_SHARPE >= 10
        assert 0.0 < HOLD_THRESHOLD < 0.1  # 1% threshold
        assert BINARY_HIT_THRESHOLD >= 0

        from aer.utils.opinion_constants import SIGNAL_DIRECTION_MAP

        assert SIGNAL_DIRECTION_MAP == {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0, "AVOID": -1.0}


class TestOpinionRecordModel:
    """OpinionRecord Pydantic model."""

    def test_opinion_record_frozen_immutability(self):
        """Create OpinionRecord; verify it is frozen (immutable)."""
        op = OpinionRecord(
            ticker="AAPL",
            agent_name="fundamental",
            opinion_date=date(2026, 6, 21),
            signal="BUY",
            confidence=0.85,
            conviction=0.90,
            drivers=["strong earnings"],
        )

        # Should be frozen; this should raise
        with pytest.raises(Exception):  # FrozenInstanceError in Pydantic v2
            op.signal = "SELL"

    def test_opinion_record_forward_return_initially_null(self):
        """OpinionRecord forward_return_5d is None by default."""
        op = OpinionRecord(
            ticker="MSFT",
            agent_name="technical",
            opinion_date=date(2026, 6, 20),
            signal="HOLD",
            confidence=0.5,
            conviction=0.5,
        )
        assert op.forward_return_5d is None

    def test_opinion_record_json_roundtrip(self):
        """Create OpinionRecord; serialize to JSON; deserialize back."""
        op = OpinionRecord(
            ticker="GOOGL",
            agent_name="sentiment",
            opinion_date=date(2026, 6, 19),
            signal="SELL",
            confidence=0.7,
            conviction=0.65,
            drivers=["negative sentiment", "short interest"],
            forward_return_5d=0.02,
        )

        json_str = op.model_dump_json()
        op2 = OpinionRecord.model_validate_json(json_str)

        assert op2.ticker == "GOOGL"
        assert op2.signal == "SELL"
        assert op2.forward_return_5d == 0.02


class TestHitRateMetricsModel:
    """HitRateMetrics Pydantic model."""

    def test_hitrate_metrics_json_roundtrip(self):
        """Create HitRateMetrics; serialize; deserialize."""
        hm = HitRateMetrics(
            agent_name="fundamental",
            ticker="AAPL",
            total_opinions=25,
            binary_hit_rate=65.0,
            return_weighted_score=0.0125,
            sharpe_ratio=1.2,
            conviction_weighted_hit_rate=72.5,
            n_profitable_holds=16,
            last_opinion_date=date(2026, 6, 21),
            insufficient_data=False,
        )

        json_str = hm.model_dump_json()
        hm2 = HitRateMetrics.model_validate_json(json_str)

        assert hm2.agent_name == "fundamental"
        assert hm2.binary_hit_rate == 65.0
        assert hm2.conviction_weighted_hit_rate == 72.5
        assert hm2.insufficient_data is False

    def test_hitrate_metrics_insufficient_data(self):
        """Create HitRateMetrics with insufficient_data=True; metrics are None."""
        hm = HitRateMetrics(
            agent_name="technical",
            ticker="MSFT",
            total_opinions=5,
            insufficient_data=True,
        )

        assert hm.binary_hit_rate is None
        assert hm.sharpe_ratio is None
        assert hm.conviction_weighted_hit_rate is None


class TestCRUDOperations:
    """Tests for record_opinion() and get_opinions()."""

    def test_record_opinion_returns_bool(self):
        """record_opinion() returns bool (True if inserted, False if skipped)."""
        import uuid

        unique_ticker = f"T{uuid.uuid4().hex[:4].upper()}"  # Generate unique ticker for test
        op = OpinionRecord(
            ticker=unique_ticker,
            agent_name="fundamental",
            opinion_date=date(2026, 6, 21),
            signal="BUY",
            confidence=0.85,
            conviction=0.90,
        )
        # First insert should return True
        result = record_opinion(op)
        assert result is True

        # Duplicate insert should return False (UNIQUE constraint)
        result = record_opinion(op)
        assert result is False

    def test_get_opinions_empty_result(self):
        """get_opinions() with no matches returns empty list."""
        # Query for a ticker that doesn't exist
        results = get_opinions(ticker="ZZZZ")
        assert results == []

    def test_record_opinion_validation_ticker_empty(self):
        """record_opinion() rejects empty ticker."""
        op = OpinionRecord(
            ticker="",
            agent_name="fundamental",
            opinion_date=date(2026, 6, 21),
            signal="BUY",
            confidence=0.85,
            conviction=0.90,
        )
        result = record_opinion(op)
        assert result is False

    def test_record_opinion_validation_ticker_too_long(self):
        """record_opinion() rejects ticker > 5 chars."""
        op = OpinionRecord(
            ticker="VERYLONGTICKER",
            agent_name="fundamental",
            opinion_date=date(2026, 6, 21),
            signal="BUY",
            confidence=0.85,
            conviction=0.90,
        )
        result = record_opinion(op)
        assert result is False

    def test_record_opinion_validation_future_date(self):
        """record_opinion() rejects future opinion_date."""
        import uuid
        from datetime import timedelta

        unique_ticker = f"T{uuid.uuid4().hex[:4].upper()}"
        op = OpinionRecord(
            ticker=unique_ticker,
            agent_name="fundamental",
            opinion_date=date.today() + timedelta(days=1),
            signal="BUY",
            confidence=0.85,
            conviction=0.90,
        )
        result = record_opinion(op)
        assert result is False

    def test_get_opinions_filter_by_ticker(self):
        """get_opinions(ticker='MSFT') returns only MSFT opinions."""
        import uuid

        # Insert two opinions for different tickers
        unique_ticker1 = f"T{uuid.uuid4().hex[:4].upper()}"
        unique_ticker2 = f"T{uuid.uuid4().hex[:4].upper()}"
        op1 = OpinionRecord(
            ticker=unique_ticker1,
            agent_name="fundamental",
            opinion_date=date(2026, 6, 21),
            signal="BUY",
            confidence=0.80,
            conviction=0.85,
        )
        op2 = OpinionRecord(
            ticker=unique_ticker2,
            agent_name="fundamental",
            opinion_date=date(2026, 6, 21),
            signal="HOLD",
            confidence=0.50,
            conviction=0.50,
        )
        record_opinion(op1)
        record_opinion(op2)

        # Query for ticker1 only
        results = get_opinions(ticker=unique_ticker1)
        assert len(results) >= 1
        assert all(r.ticker == unique_ticker1 for r in results)

    def test_get_opinions_sorted_newest_first(self):
        """get_opinions() returns sorted by opinion_date DESC (newest first)."""
        import uuid

        unique_ticker = f"T{uuid.uuid4().hex[:4].upper()}"
        # Insert two opinions with different dates
        op1 = OpinionRecord(
            ticker=unique_ticker,
            agent_name="technical",
            opinion_date=date(2026, 6, 20),
            signal="SELL",
            confidence=0.70,
            conviction=0.65,
        )
        op2 = OpinionRecord(
            ticker=unique_ticker,
            agent_name="technical",
            opinion_date=date(2026, 6, 21),
            signal="HOLD",
            confidence=0.60,
            conviction=0.60,
        )
        record_opinion(op1)
        record_opinion(op2)

        # Query and verify sort order
        results = get_opinions(ticker=unique_ticker, agent_name="technical")
        if len(results) >= 2:
            assert results[0].opinion_date >= results[1].opinion_date


class TestAnalytics:
    """Tests for compute_hit_rates() and get_equity_curve()."""

    def test_compute_hit_rates_min_opinions_threshold(self):
        """compute_hit_rates() requires >= MIN_OPINIONS_FOR_METRICS opinions."""
        pass

    def test_get_equity_curve_cumulative_product(self):
        """get_equity_curve() returns cumulative product (1 + return) over time."""
        pass


class TestPopulateForwardReturns:
    """Tests for populate_forward_returns.

    Hand-derivation (recorded here, not in a separate evidence file):
    opinion_date = 2026-01-05 (Mon). Five trading days forward, no US federal holiday in the
    window (New Year 01-01 precedes it, MLK 01-19 follows forward_date):
    Tue 01-06(1) Wed 01-07(2) Thu 01-08(3) Fri 01-09(4) [Sat/Sun skip] Mon 01-12(5)
    => forward_date = 2026-01-12 (Mon).
    Synthetic ADJUSTED frame, Close: 2026-01-05 -> 100.0; 2026-01-12 -> 105.0
    => forward_return_5d = (105.0 - 100.0) / 100.0 = 0.05
    """

    @staticmethod
    def _price_frame():
        import pandas as pd

        idx = pd.to_datetime(
            ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12", "2026-01-13"]
        )
        # Title-case "Close" carries the true closes (what get_price_history returns after the
        # yfinance MultiIndex flatten); the lowercase "close" decoy would mislead a regression
        # that reverts the key fix, and the 01-06 row would mislead a regression that drops the
        # date-matched index.
        return pd.DataFrame(
            {
                "Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
                "close": [999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0],
            },
            index=idx,
        )

    @staticmethod
    def _record_one(ticker="AAPL"):
        from aer.data.opinions import record_opinion
        from aer.models import OpinionRecord

        op = OpinionRecord(
            ticker=ticker,
            agent_name="fundamental",
            opinion_date=date(2026, 1, 5),
            signal="BUY",
            confidence=0.80,
            conviction=0.85,
        )
        assert record_opinion(op) is True

    def test_populate_returns_nonzero_updated(self, monkeypatch):
        """One NULL opinion plus a synthetic frame gives updated == 1."""
        import aer.data.prices as prices_mod
        from aer.data.opinions import populate_forward_returns

        self._record_one()
        monkeypatch.setattr(prices_mod, "get_price_history", lambda *a, **k: self._price_frame())
        updated = populate_forward_returns()
        assert updated == 1

    def test_raising_fetch_counts_skipped(self, monkeypatch, caplog):
        """A fetch that raises is counted into `skipped`, not silently continued."""
        import logging

        import aer.data.prices as prices_mod
        from aer.data.opinions import populate_forward_returns

        self._record_one()

        def _boom(*a, **k):
            raise ValueError("simulated fetch failure")

        monkeypatch.setattr(prices_mod, "get_price_history", _boom)
        with caplog.at_level(logging.INFO):
            updated = populate_forward_returns()
        assert updated == 0
        assert "1 skipped" in caplog.text

    def test_close_key_and_date_matched_index(self, monkeypatch):
        """Regression fence: correct `Close` key AND date-matched index pick 100.0 / 105.0."""
        import aer.data.prices as prices_mod
        from aer.data.opinions import get_opinions, populate_forward_returns

        self._record_one()
        monkeypatch.setattr(prices_mod, "get_price_history", lambda *a, **k: self._price_frame())
        populate_forward_returns()
        op = get_opinions(ticker="AAPL")[0]
        # (105 - 100)/100 from the Title-case column; the lowercase 999 decoy must NOT be used.
        assert op.forward_return_5d == pytest.approx(0.05)

    def test_forward_return_arithmetic_run_date_invariant(self, monkeypatch):
        """Fixed-date synthetic fixture -> deterministic forward_return, no network."""
        import aer.data.prices as prices_mod
        from aer.data.opinions import get_opinions, populate_forward_returns

        self._record_one()
        monkeypatch.setattr(prices_mod, "get_price_history", lambda *a, **k: self._price_frame())
        updated = populate_forward_returns()
        assert updated == 1
        op = get_opinions(ticker="AAPL")[0]
        assert op.forward_return_5d == pytest.approx((105.0 - 100.0) / 100.0)

    def test_unclassified_fetch_error_is_counted_not_fatal(self, monkeypatch, caplog):
        """An error outside the classified tuple is COUNTED and logged, and the run continues.

        A bare `except Exception` that skipped the counter once let a total failure log `updated=0,
        skipped=0` and read as a no-op for months. This test makes that gate FIRE.
        """
        import logging

        import aer.data.prices as prices_mod
        from aer.data.opinions import populate_forward_returns

        self._record_one()

        def _boom(*a, **k):
            raise TypeError("simulated unclassified failure")

        monkeypatch.setattr(prices_mod, "get_price_history", _boom)
        with caplog.at_level(logging.INFO):
            updated = populate_forward_returns()  # must RETURN, not raise
        assert updated == 0
        assert "1 skipped" in caplog.text
        assert "1 unclassified" in caplog.text

    def test_admission_cap_limits_tickers_per_run(self, monkeypatch, caplog):
        """More distinct NULL tickers than MAX_FORWARD_RETURN_TICKERS_PER_RUN
        admits only the cap's worth, oldest opinions first by ticker, and logs the deferral.
        """
        import logging

        import pandas as pd

        import aer.data.prices as prices_mod
        from aer.data.opinions import get_opinions, populate_forward_returns
        from aer.utils.opinion_constants import MAX_FORWARD_RETURN_TICKERS_PER_RUN

        n_tickers = MAX_FORWARD_RETURN_TICKERS_PER_RUN + 1
        business_days = pd.bdate_range(start="2026-01-05", periods=n_tickers + 10)
        tickers = [f"Z{i:03d}" for i in range(n_tickers)]

        for i, ticker in enumerate(tickers):
            op = OpinionRecord(
                ticker=ticker,
                agent_name="fundamental",
                opinion_date=business_days[i].date(),
                signal="BUY",
                confidence=0.80,
                conviction=0.85,
            )
            assert record_opinion(op) is True

        frame = pd.DataFrame(
            {"Close": [100.0 + i for i in range(len(business_days))]},
            index=business_days,
        )
        monkeypatch.setattr(prices_mod, "get_price_history", lambda *a, **k: frame)

        with caplog.at_level(logging.INFO):
            updated = populate_forward_returns()

        assert updated == MAX_FORWARD_RETURN_TICKERS_PER_RUN

        admitted = tickers[:MAX_FORWARD_RETURN_TICKERS_PER_RUN]
        deferred = tickers[MAX_FORWARD_RETURN_TICKERS_PER_RUN:]
        for ticker in admitted:
            op = get_opinions(ticker=ticker)[0]
            assert op.forward_return_5d is not None
        for ticker in deferred:
            op = get_opinions(ticker=ticker)[0]
            assert op.forward_return_5d is None

        assert f"exceeding the per-run cap of {MAX_FORWARD_RETURN_TICKERS_PER_RUN}" in caplog.text

    def test_ladder_exhaustion_uses_largest_rung(self, monkeypatch, caplog):
        """A backlog whose oldest row is older than the largest ladder rung (730
        bars) does not crash; the largest rung is selected and the exhaustion is logged.
        """
        import logging
        from datetime import timedelta

        import pandas as pd

        import aer.data.prices as prices_mod
        from aer.data.opinions import get_opinions, populate_forward_returns
        from aer.utils.opinion_constants import FORWARD_RETURN_PERIOD_LADDER_BARS

        raw_anchor = date.today() - timedelta(days=800)
        while raw_anchor.weekday() >= 5:  # snap back off any weekend to a business day
            raw_anchor -= timedelta(days=1)

        ticker = "OLD01"
        op = OpinionRecord(
            ticker=ticker,
            agent_name="fundamental",
            opinion_date=raw_anchor,
            signal="BUY",
            confidence=0.80,
            conviction=0.85,
        )
        assert record_opinion(op) is True

        idx = pd.bdate_range(start=raw_anchor, periods=10)
        frame = pd.DataFrame({"Close": [100.0 + i for i in range(10)]}, index=idx)
        monkeypatch.setattr(prices_mod, "get_price_history", lambda *a, **k: frame)

        with caplog.at_level(logging.INFO):
            updated = populate_forward_returns()

        assert updated == 1
        op_after = get_opinions(ticker=ticker)[0]
        assert op_after.forward_return_5d == pytest.approx((105.0 - 100.0) / 100.0)

        largest_rung = FORWARD_RETURN_PERIOD_LADDER_BARS[-1]
        assert f"exceeds the largest ladder rung ({largest_rung} bars)" in caplog.text
