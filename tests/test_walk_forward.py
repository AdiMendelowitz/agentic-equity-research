"""Walk-forward module: constants, the fold layout as computed by the code, and result serialisation.

Behaviour of run_walk_forward itself (cutoff, insufficient history, signal failures, no lookahead) is tested
hermetically in tests/test_integration_walkforward.py.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from aer.risk.models import FoldResult, WalkForwardResult
from aer.risk.walkforward import _compute_fold_boundaries
from aer.utils.wfo_constants import TOTAL_MIN_BARS, WFO_MIN_IS_BARS, WFO_N_FOLDS, WFO_OOS_SIZE_BARS, WFO_PURGE_GAP_BARS


class TestWFOConstants:
    def test_wfo_constants_boundaries(self) -> None:
        assert WFO_N_FOLDS == 5
        assert WFO_PURGE_GAP_BARS == 10
        assert WFO_OOS_SIZE_BARS == 55
        assert WFO_MIN_IS_BARS == 252
        assert TOTAL_MIN_BARS == WFO_MIN_IS_BARS + WFO_N_FOLDS * (WFO_PURGE_GAP_BARS + WFO_OOS_SIZE_BARS) == 577


class TestFoldGeometry:
    """The layout _compute_fold_boundaries returns: expanding IS from bar 0, anchored to the last bar."""

    def test_minimum_history_layout(self) -> None:
        folds = _compute_fold_boundaries(TOTAL_MIN_BARS)
        assert folds[0] == (0, 252, 262, 317)
        assert folds[1] == (0, 317, 327, 382)
        assert folds[4] == (0, 512, 522, 577)

    @pytest.mark.parametrize("num_bars", [TOTAL_MIN_BARS, TOTAL_MIN_BARS + 1, TOTAL_MIN_BARS + 423])
    def test_layout_invariants(self, num_bars: int) -> None:
        folds = _compute_fold_boundaries(num_bars)
        assert len(folds) == WFO_N_FOLDS
        assert folds[-1][3] == num_bars
        assert folds[0][1] == num_bars - TOTAL_MIN_BARS + WFO_MIN_IS_BARS
        for is_start, is_end, oos_start, oos_end in folds:
            assert is_start == 0
            assert oos_start - is_end == WFO_PURGE_GAP_BARS
            assert oos_end - oos_start == WFO_OOS_SIZE_BARS
        for earlier, later in zip(folds, folds[1:], strict=False):
            assert later[1] == earlier[3]

    def test_short_history_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _compute_fold_boundaries(TOTAL_MIN_BARS - 1)


def _fold(fold_idx: int = 0, fold_error: str | None = None) -> FoldResult:
    ok = fold_error is None
    return FoldResult(
        fold_idx=fold_idx,
        is_start=date(2023, 1, 1),
        is_end=date(2023, 12, 31),
        n_is_bars=252,
        oos_start=date(2024, 2, 10),
        oos_end=date(2024, 4, 5),
        n_oos_bars=55,
        oos_sharpe_long=0.85 if ok else None,
        oos_sharpe_ls=1.12 if ok else None,
        oos_drawdown=-0.08 if ok else None,
        factor_alpha=0.03 if ok else None,
        factor_alpha_significant=ok,
        fold_error=fold_error,
    )


class TestResultSerialization:
    def test_fold_result_json_roundtrip(self) -> None:
        for fold in (_fold(), _fold(2, "RuntimeError: signal generation failed")):
            assert FoldResult.model_validate_json(fold.model_dump_json()) == fold

    def test_walk_forward_result_json_roundtrip(self) -> None:
        wfr = WalkForwardResult(
            ticker="AAPL",
            as_of_date=date(2026, 6, 19),
            folds=[_fold(), _fold(2, "RuntimeError: signal generation failed")],
            mean_oos_sharpe_long=0.85,
            mean_oos_sharpe_ls=1.12,
            mean_oos_drawdown=-0.08,
            n_folds_completed=1,
            n_folds_with_factor_alpha=1,
            insufficient_history=False,
        )
        assert WalkForwardResult.model_validate_json(wfr.model_dump_json()) == wfr

    @pytest.mark.parametrize(
        "override",
        [{"n_is_bars": 251}, {"n_oos_bars": 54}, {"n_oos_bars": 56}, {"fold_idx": -1}, {"fold_idx": 5}],
    )
    def test_out_of_range_geometry_is_rejected(self, override: dict[str, int]) -> None:
        fields = _fold().model_dump() | override
        with pytest.raises(ValidationError):
            FoldResult(**fields)


class TestInvestmentBriefWalkForwardIntegration:
    def test_investment_brief_walk_forward_field(self) -> None:
        from aer.models import InvestmentBrief

        brief = InvestmentBrief(ticker="AAPL", company_name="Apple Inc.")
        assert brief.walk_forward is None

    def test_investment_brief_accepts_walk_forward_result(self) -> None:
        from aer.models import InvestmentBrief

        wfr = WalkForwardResult(ticker="AAPL", as_of_date=date(2026, 6, 19), folds=[], insufficient_history=True)
        brief = InvestmentBrief(ticker="AAPL", company_name="Apple Inc.").model_copy(update={"walk_forward": wfr})
        assert brief.walk_forward is not None
        assert brief.walk_forward.insufficient_history is True
