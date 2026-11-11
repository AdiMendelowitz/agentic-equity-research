"""Basic coverage for src/utils/metrics.py — pure functions, no I/O."""

import pytest

from aer.utils.metrics import calmar_ratio, hit_rate, max_drawdown, profit_factor


def _rec(correct: bool | None = None, ret: float | None = None, signal: str = "bullish") -> dict:
    return {"outcome_direction_correct": correct, "outcome_return_pct": ret, "signal": signal}


# --- hit_rate ---


def test_hit_rate_all_correct() -> None:
    records = [_rec(correct=True), _rec(correct=True), _rec(correct=True)]
    assert hit_rate(records) == 1.0


def test_hit_rate_half_correct() -> None:
    records = [_rec(correct=True), _rec(correct=False)]
    assert hit_rate(records) == pytest.approx(0.5)


def test_hit_rate_empty() -> None:
    assert hit_rate([]) == 0.0


def test_hit_rate_unlabeled_excluded() -> None:
    records = [_rec(correct=True), _rec(correct=None)]
    assert hit_rate(records) == pytest.approx(1.0)


# --- profit_factor ---


def test_profit_factor_positive() -> None:
    records = [
        _rec(ret=0.10),
        _rec(ret=0.05),
        _rec(ret=-0.02),
    ]
    result = profit_factor(records)
    assert result > 1.0
    assert result == pytest.approx(0.15 / 0.02)


def test_profit_factor_no_losses() -> None:
    records = [_rec(ret=0.05), _rec(ret=0.10)]
    assert profit_factor(records) == float("inf")


def test_profit_factor_no_wins_no_losses() -> None:
    assert profit_factor([]) == 0.0


# --- max_drawdown ---


def test_max_drawdown_basic() -> None:
    records = [
        _rec(ret=0.10),  # cumulative 0.10, peak 0.10
        _rec(ret=0.05),  # cumulative 0.15, peak 0.15
        _rec(ret=-0.10),  # cumulative 0.05, dd = 0.10
        _rec(ret=-0.08),  # cumulative -0.03, dd = 0.18  <- max
        _rec(ret=0.20),  # cumulative 0.17, peak stays 0.17 (< 0.15 prev peak 0.15... wait)
    ]
    # Manual: peak sequence: 0.10, 0.15, 0.15, 0.15, 0.17
    # dd sequence:           0.0,  0.0,  0.10, 0.18, 0.0 (0.17-0.17=0)
    # Actually after ret=0.20: cumulative=0.17, new peak=0.17, dd=0
    assert max_drawdown(records) == pytest.approx(0.18)


def test_max_drawdown_empty() -> None:
    assert max_drawdown([]) == 0.0


def test_max_drawdown_all_positive() -> None:
    records = [_rec(ret=0.05), _rec(ret=0.03), _rec(ret=0.02)]
    assert max_drawdown(records) == pytest.approx(0.0)


# --- calmar_ratio ---


def test_calmar_ratio_zero_drawdown() -> None:
    records = [_rec(ret=0.01), _rec(ret=0.02), _rec(ret=0.03)]
    assert calmar_ratio(records) == 0.0


def test_calmar_ratio_positive() -> None:
    records = [
        _rec(ret=0.10),
        _rec(ret=-0.05),
        _rec(ret=0.08),
    ]
    dd = max_drawdown(records)
    n = 3
    annualized = (sum([0.10, -0.05, 0.08]) / n) * 252
    expected = annualized / dd
    assert calmar_ratio(records) == pytest.approx(expected)


def test_calmar_ratio_empty() -> None:
    assert calmar_ratio([]) == 0.0
