"""Pydantic v2 models for risk analytics."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RiskMetrics(BaseModel):
    """Computed risk analytics for a price return series.

    VaR/CVaR fields follow loss convention: values are non-positive decimals
    (e.g. -0.032 = 3.2% loss). None indicates insufficient history.
    downside_dev is a positive annualized standard deviation of negative returns.
    """

    # Drawdown detail
    dd_start: date | None = None  # date of drawdown peak (start of impairment)
    dd_end: date | None = None  # date of trough (deepest loss point)
    recovery_days: int | None = None  # trough → recovery: days to recover from trough
    dd_duration_days: int | None = None  # peak → recovery: total impairment duration

    # Downside deviation (MAR=0, annualized, positive float = annualized std of negative returns)
    downside_dev: float | None = None

    # Historical VaR/CVaR (empirical quantile - from empyrical or numpy fallback)
    var_95_hist: float | None = None  # negative decimal: 5th percentile of daily returns
    cvar_95_hist: float | None = None  # negative decimal: expected loss in worst 5% of days
    var_99_hist: float | None = None  # None when history < MIN_VAR_99_POINTS (250)
    cvar_99_hist: float | None = None  # None when history < MIN_VAR_99_POINTS (250)

    # Parametric VaR/CVaR (Gaussian normal distribution assumption)
    var_95_param: float | None = None
    cvar_95_param: float | None = None
    var_99_param: float | None = None  # None when history < MIN_VAR_99_POINTS (250)
    cvar_99_param: float | None = None  # None when history < MIN_VAR_99_POINTS (250)

    # Guards
    insufficient_history: bool = False  # True when N < MIN_RISK_POINTS (30)
    insufficient_var95: bool = False  # True when N < MIN_VAR_95_POINTS (60)
    insufficient_var99: bool = False  # True when N < MIN_VAR_99_POINTS (250)

    @field_validator(
        "var_95_hist",
        "cvar_95_hist",
        "var_99_hist",
        "cvar_99_hist",
        "var_95_param",
        "cvar_95_param",
        "var_99_param",
        "cvar_99_param",
        mode="before",
    )
    @classmethod
    def must_be_nonpositive(cls, v: float | None) -> float | None:
        if v is not None and v > 0:
            raise ValueError(f"VaR/CVaR must be non-positive (loss convention), got {v}")
        return v

    model_config = ConfigDict(frozen=True)


class FactorLoadings(BaseModel):
    """Fama-French 5-factor OLS regression output for a single ticker.

    alpha is annualised via geometric compounding: (1 + monthly_intercept)^12 - 1.
    All beta fields (mkt_rf..cma) are raw monthly OLS coefficients.
    t_stats / p_values use HAC (Newey-West, 3 lags) standard errors.
    t_stats / p_values keys: "alpha", "mkt_rf", "smb", "hml", "rmw", "cma" (OLS "const" mapped to "alpha").
    """

    alpha: float | None = None  # Jensen's alpha, annualised via (1+monthly)^12-1, decimal
    mkt_rf: float | None = None  # Market beta (FF5), raw monthly OLS coefficient
    smb: float | None = None  # Size factor beta (FF5), raw monthly OLS coefficient
    hml: float | None = None  # Value factor beta (FF5), raw monthly OLS coefficient
    rmw: float | None = None  # Profitability factor beta (FF5), raw monthly OLS coefficient
    cma: float | None = None  # Investment factor beta (FF5), raw monthly OLS coefficient
    r_squared: float | None = None  # OLS R² (0–1); typical range 0.20–0.60 for single equities
    t_stats: dict[str, float] | None = None  # HAC t-statistics; keys: "alpha","mkt_rf","smb","hml","rmw","cma"
    p_values: dict[str, float] | None = None  # HAC p-values; same 6 keys as t_stats
    n_obs: int = 0  # Monthly observations used in OLS regression after NaN removal
    insufficient_history: bool = False  # True when aligned months < FACTOR_MIN_MONTHS (36)
    low_data_quality: bool = False  # True when 36 <= aligned months < FACTOR_PREFERRED_MONTHS (60)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_stat_key_consistency(self) -> FactorLoadings:
        expected_keys = {"alpha", "mkt_rf", "smb", "hml", "rmw", "cma"}
        if self.t_stats is not None and set(self.t_stats.keys()) != expected_keys:
            raise ValueError(f"t_stats keys must be exactly {expected_keys}, got {set(self.t_stats.keys())}")
        if self.p_values is not None and set(self.p_values.keys()) != expected_keys:
            raise ValueError(f"p_values keys must be exactly {expected_keys}, got {set(self.p_values.keys())}")
        if (self.t_stats is None) != (self.p_values is None):
            raise ValueError("t_stats and p_values must both be None or both be populated")
        return self


# ---------------------------------------------------------------------------
# Walk-Forward Optimization (WFO) Results
# ---------------------------------------------------------------------------


class FoldResult(BaseModel):
    """Results for a single walk-forward optimization fold.

    Holds in-sample parameters and out-of-sample performance for one fold.
    All metric fields (oos_sharpe_long, oos_sharpe_ls, oos_drawdown, factor_alpha)
    are None if:
    - The parent WalkForwardResult has insufficient_history=True (run-level), OR
    - This specific fold raised an exception during computation (fold_error is set)

    fold_error contains the exception message if computation failed; None if success.
    """

    fold_idx: int = Field(..., ge=0, le=4, description="0-indexed fold number (0–4 for 5-fold WFO)")
    is_start: date = Field(..., description="Start date of in-sample window")
    is_end: date = Field(..., description="End date of in-sample window")
    n_is_bars: int = Field(..., ge=252, description="Number of in-sample bars for this fold")
    oos_start: date = Field(..., description="Start date of out-of-sample window")
    oos_end: date = Field(..., description="End date of out-of-sample window")
    n_oos_bars: int = Field(..., ge=55, le=55, description="Number of OOS bars in this fold (always 55)")
    oos_sharpe_long: float | None = Field(None, description="Sharpe ratio on OOS long-only signal returns")
    oos_sharpe_ls: float | None = Field(
        None, description="Sharpe ratio on OOS long-short signal returns (assumes zero transaction costs)"
    )
    oos_drawdown: float | None = Field(None, description="Max drawdown on OOS returns (empyrical.max_drawdown)")
    factor_alpha: float | None = Field(None, description="Fama-French 5-factor Jensen's alpha, annualized")
    factor_alpha_significant: bool = Field(False, description="True if factor_alpha p-value < 0.05")
    fold_error: str | None = Field(None, description="Exception message if this fold failed; None if success")

    model_config = ConfigDict(frozen=True)


class WalkForwardResult(BaseModel):
    """Aggregated walk-forward optimization result for one ticker as-of date.

    Computed across 5 non-overlapping folds. If insufficient_history=True,
    all metric fields in all folds are None, and the fold list is populated
    with fold geometry but no performance data (fold_error is set).

    version field allows traceability of schema changes; bumped when fold fields change.
    """

    ticker: str = Field(..., description="Stock ticker symbol")
    as_of_date: date = Field(..., description="Reference date for this walk-forward (typically brief date)")
    folds: list[FoldResult] = Field(default_factory=list, description="List of 5 FoldResult objects")
    mean_oos_sharpe_long: float | None = Field(None, description="Mean OOS Sharpe (long-only) across folds")
    mean_oos_sharpe_ls: float | None = Field(None, description="Mean OOS Sharpe (long-short) across folds")
    std_oos_sharpe_long: float | None = Field(None, description="Std OOS Sharpe (long-only) across folds")
    std_oos_sharpe_ls: float | None = Field(None, description="Std OOS Sharpe (long-short) across folds")
    mean_oos_drawdown: float | None = Field(None, description="Mean max drawdown across folds")
    n_folds_completed: int = Field(0, ge=0, le=5, description="Number of folds that completed without error")
    n_folds_with_factor_alpha: int = Field(0, ge=0, le=5, description="Number of folds where factor_alpha is not None")
    insufficient_history: bool = Field(False, description="True if history < 577 bars (TOTAL_MIN_BARS)")
    version: str = Field("1.0", description="Schema version for backward compatibility")

    model_config = ConfigDict(frozen=True)
