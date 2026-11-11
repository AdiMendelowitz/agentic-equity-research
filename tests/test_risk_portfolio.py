"""Tests for src/risk/portfolio.py — compute_risk and compute_drawdown_detail.

Covers:
- Tiered sample guards (MIN_RISK_POINTS=30, MIN_VAR_95_POINTS=60, MIN_VAR_99_POINTS=250)
- VaR sign convention (non-positive, loss convention)
- Downside deviation uses N_total denominator
- recovery_days = trough→recovery; dd_duration_days = peak→recovery
- No imports from aer.portfolio.* (circular dependency absent)
"""

from __future__ import annotations

import importlib
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from aer.risk.models import RiskMetrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _make_returns(n: int, rng: np.random.Generator) -> pd.Series:
    """Synthetic daily returns with DatetimeIndex."""
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    vals = rng.normal(0.0005, 0.015, n)
    return pd.Series(vals, index=dates)


# ---------------------------------------------------------------------------
# Import isolation — no circular dependency
# ---------------------------------------------------------------------------


def test_no_portfolio_imports() -> None:
    """portfolio.py must not import from aer.portfolio.*."""
    mod = importlib.import_module("aer.risk.portfolio")
    src = inspect.getsource(mod)
    assert "from aer.portfolio" not in src, "Circular import: from aer.portfolio.* found"
    assert "import aer.portfolio" not in src, "Circular import: import aer.portfolio found"


def test_compute_risk_importable() -> None:
    from aer.risk.portfolio import compute_drawdown_detail, compute_risk  # noqa: F401

    assert callable(compute_risk)
    assert callable(compute_drawdown_detail)


# ---------------------------------------------------------------------------
# Tier 1: N < 30 — insufficient_history
# ---------------------------------------------------------------------------


def test_insufficient_history_returns_guard_model(rng: np.random.Generator) -> None:
    r = _make_returns(25, rng)
    from aer.risk.portfolio import compute_risk

    result = compute_risk(r)
    assert result.insufficient_history is True
    assert result.var_95_hist is None
    assert result.var_99_hist is None
    assert result.downside_dev is None
    assert result.dd_start is None


def test_exactly_29_returns_insufficient(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(29, rng))
    assert result.insufficient_history is True


def test_exactly_30_returns_sufficient_for_risk(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(30, rng))
    assert result.insufficient_history is False


# ---------------------------------------------------------------------------
# Tier 2: 30 <= N < 60 — insufficient_var95
# ---------------------------------------------------------------------------


def test_n50_var95_none(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(50, rng))
    assert result.insufficient_history is False
    assert result.insufficient_var95 is True
    assert result.var_95_hist is None
    assert result.cvar_95_hist is None
    assert result.var_99_hist is None


def test_exactly_59_insufficient_var95(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(59, rng))
    assert result.insufficient_var95 is True


def test_exactly_60_sufficient_for_var95(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(60, rng))
    assert result.insufficient_var95 is False
    assert result.var_95_hist is not None


# ---------------------------------------------------------------------------
# Tier 3: 60 <= N < 250 — var95 present, var99 None
# ---------------------------------------------------------------------------


def test_n80_var95_present_var99_none(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(80, rng))
    assert result.var_95_hist is not None
    assert result.cvar_95_hist is not None
    assert result.insufficient_var99 is True
    assert result.var_99_hist is None
    assert result.cvar_99_hist is None


def test_exactly_249_insufficient_var99(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(249, rng))
    assert result.insufficient_var99 is True
    assert result.var_99_hist is None


# ---------------------------------------------------------------------------
# Tier 4: N >= 250 — all fields populated
# ---------------------------------------------------------------------------


def test_n300_all_fields_populated(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(300, rng))
    assert result.insufficient_history is False
    assert result.insufficient_var95 is False
    assert result.insufficient_var99 is False
    assert result.var_95_hist is not None
    assert result.var_99_hist is not None
    assert result.cvar_95_hist is not None
    assert result.cvar_99_hist is not None
    assert result.downside_dev is not None


# ---------------------------------------------------------------------------
# VaR sign convention — loss convention (non-positive)
# ---------------------------------------------------------------------------


def test_var95_hist_is_nonpositive(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(300, rng))
    assert result.var_95_hist <= 0, "VaR 95% must be <= 0 (loss convention)"


def test_var99_hist_is_nonpositive(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(300, rng))
    assert result.var_99_hist <= 0, "VaR 99% must be <= 0 (loss convention)"


def test_var99_worse_than_var95(rng: np.random.Generator) -> None:
    """VaR 99% should have a larger (more negative) loss than VaR 95%."""
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(300, rng))
    assert (
        result.var_99_hist <= result.var_95_hist
    ), f"var_99_hist ({result.var_99_hist}) must be <= var_95_hist ({result.var_95_hist})"


def test_cvar95_worse_than_var95(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(300, rng))
    assert (
        result.cvar_95_hist <= result.var_95_hist
    ), f"cvar_95_hist ({result.cvar_95_hist}) must be <= var_95_hist ({result.var_95_hist})"


# ---------------------------------------------------------------------------
# Downside deviation — positive, uses N_total denominator
# ---------------------------------------------------------------------------


def test_downside_dev_positive(rng: np.random.Generator) -> None:
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(300, rng))
    assert result.downside_dev is not None
    assert result.downside_dev > 0


def test_downside_dev_uses_n_total_denominator() -> None:
    """With N=4 and 2 negative returns, downside_dev uses denominator=4, not 2."""
    from aer.risk.portfolio import compute_risk

    # Build a 30-element series where we control the composition
    neg = np.array([-0.05, -0.03])
    pos = np.array([0.01, 0.02] * 14)
    all_vals = np.concatenate([neg, pos])[:30]
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    r = pd.Series(all_vals, index=dates)

    result = compute_risk(r)
    if result.insufficient_var95:
        # expected when N=30 — just check downside_dev is set
        assert result.downside_dev is not None or all_vals[all_vals < 0].size == 0

    # Direct computation with a larger series ensuring we can compare
    n = 300
    np_rng = np.random.default_rng(1)
    all_large = np_rng.normal(0.0005, 0.015, n)
    neg_mask = all_large < 0
    neg_sq_sum = float((all_large[neg_mask] ** 2).sum())
    expected_dd = math.sqrt(252) * math.sqrt(neg_sq_sum / n)

    r_large = pd.Series(all_large, index=pd.date_range("2020-01-02", periods=n, freq="B"))
    result_large = compute_risk(r_large)
    assert result_large.downside_dev is not None
    assert (
        abs(result_large.downside_dev - round(expected_dd, 6)) < 1e-9
    ), f"downside_dev={result_large.downside_dev}, expected={round(expected_dd, 6)}"


def test_downside_dev_none_when_no_negative_returns() -> None:
    """If all returns are non-negative, downside_dev must be None."""
    from aer.risk.portfolio import compute_risk

    # 60 positive returns to get past var95 threshold
    pos = np.linspace(0.001, 0.003, 60)
    r = pd.Series(pos, index=pd.date_range("2024-01-01", periods=60, freq="B"))
    result = compute_risk(r)
    assert result.downside_dev is None


# ---------------------------------------------------------------------------
# Drawdown detail: recovery_days vs dd_duration_days
# ---------------------------------------------------------------------------


def test_drawdown_detail_known_series() -> None:
    """Known series: peak=102 on day1, trough=95 on day3, recovery=103 on day7.

    Series: [100, 102, 98, 95, 96, 99, 101, 103]
    Peak=102 (index 1), trough=95 (index 3), recovery=103 (index 7).
    recovery_days = (day7 - day3) = 4 business days (trough -> recovery)
    dd_duration_days = (day7 - day1) = 6 business days (peak -> recovery)
    """
    from aer.risk.portfolio import compute_drawdown_detail

    dates = pd.date_range("2024-01-01", periods=8)
    vals = pd.Series(
        [100.0, 102.0, 98.0, 95.0, 96.0, 99.0, 101.0, 103.0],
        index=dates,
    )
    dd = compute_drawdown_detail(vals)
    assert dd["dd_start"] is not None
    assert dd["dd_end"] is not None

    # dd_end should be the trough (index 3: 95.0)
    trough_date = vals.idxmin().date()
    assert dd["dd_end"] == trough_date

    # recovery_days = trough -> recovery (not peak -> recovery)
    assert dd["recovery_days"] is not None
    assert dd["dd_duration_days"] is not None
    assert (
        dd["recovery_days"] < dd["dd_duration_days"]
    ), "recovery_days (trough->recovery) must be < dd_duration_days (peak->recovery)"


def test_drawdown_detail_no_recovery() -> None:
    """Series that never recovers to peak — recovery_days should be None."""
    from aer.risk.portfolio import compute_drawdown_detail

    vals = pd.Series(
        [100.0, 102.0, 98.0, 95.0, 90.0],
        index=pd.date_range("2024-01-01", periods=5),
    )
    dd = compute_drawdown_detail(vals)
    assert dd["recovery_days"] is None
    assert dd["dd_duration_days"] is None


def test_drawdown_detail_monotone_increasing() -> None:
    """No drawdown in monotone-increasing series."""
    from aer.risk.portfolio import compute_drawdown_detail

    vals = pd.Series(
        [100.0, 101.0, 102.0, 103.0],
        index=pd.date_range("2024-01-01", periods=4),
    )
    dd = compute_drawdown_detail(vals)
    assert dd["dd_start"] is None
    assert dd["dd_end"] is None
    assert dd["recovery_days"] is None
    assert dd["dd_duration_days"] is None


def test_recovery_days_trough_to_recovery(rng: np.random.Generator) -> None:
    """Verify recovery_days measures trough-to-recovery, not peak-to-recovery."""
    from aer.risk.portfolio import compute_risk

    result = compute_risk(_make_returns(300, rng))
    if result.dd_start and result.dd_end and result.recovery_days is not None and result.dd_duration_days is not None:
        # dd_duration_days (peak→recovery) >= recovery_days (trough→recovery)
        assert result.dd_duration_days >= result.recovery_days, (
            "dd_duration_days must be >= recovery_days "
            f"(got duration={result.dd_duration_days}, recovery={result.recovery_days})"
        )


# ---------------------------------------------------------------------------
# Input sanitisation
# ---------------------------------------------------------------------------


def test_duplicate_dates_handled() -> None:
    """Duplicate index dates must be deduplicated without error."""
    from aer.risk.portfolio import compute_risk

    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    # Introduce 5 duplicates
    dup_dates = dates.append(dates[:5])
    np_rng = np.random.default_rng(7)
    vals = np_rng.normal(0.0005, 0.015, len(dup_dates))
    r = pd.Series(vals, index=dup_dates)
    result = compute_risk(r)
    assert isinstance(result, RiskMetrics)


def test_nan_values_dropped() -> None:
    """NaN returns must be dropped gracefully."""
    from aer.risk.portfolio import compute_risk

    np_rng = np.random.default_rng(99)
    vals = np_rng.normal(0.0005, 0.015, 300).tolist()
    vals[10] = float("nan")
    vals[150] = float("nan")
    r = pd.Series(vals, index=pd.date_range("2020-01-02", periods=300, freq="B"))
    result = compute_risk(r)
    assert isinstance(result, RiskMetrics)
