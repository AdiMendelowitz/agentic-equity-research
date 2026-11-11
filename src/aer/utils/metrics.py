"""Metric suite — all computed on demand from labeled records. Never stored."""

from collections import Counter
from typing import Any

from aer.utils.recommendation_log import load_records


def hit_rate(records: list[dict[str, Any]]) -> float:
    """Fraction of records with outcome_direction_correct == True."""
    labeled = [r for r in records if r.get("outcome_direction_correct") is not None]
    if not labeled:
        return 0.0
    return sum(1 for r in labeled if r["outcome_direction_correct"]) / len(labeled)


def profit_factor(records: list[dict[str, Any]]) -> float:
    """Gross wins pct / gross losses pct. Returns 0.0 if no losses."""
    labeled = [r for r in records if r.get("outcome_return_pct") is not None]
    wins = sum(r["outcome_return_pct"] for r in labeled if r["outcome_return_pct"] > 0)
    losses = abs(sum(r["outcome_return_pct"] for r in labeled if r["outcome_return_pct"] < 0))
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def max_drawdown(records: list[dict[str, Any]]) -> float:
    """Max peak-to-trough on cumulative return series. Returns 0.0 for empty."""
    returns = [r["outcome_return_pct"] for r in records if r.get("outcome_return_pct") is not None]
    if not returns:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for ret in returns:
        cumulative += ret
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


def calmar_ratio(records: list[dict[str, Any]]) -> float:
    """Annualized return / max_drawdown. Returns 0.0 if drawdown is 0."""
    returns = [r["outcome_return_pct"] for r in records if r.get("outcome_return_pct") is not None]
    if not returns:
        return 0.0
    n = len(returns)
    annualized = (sum(returns) / n) * 252  # approximate trading days
    dd = max_drawdown(records)
    if dd == 0:
        return 0.0
    return annualized / dd


def regime_conditioned_hit_rate(agent_id: str, regime: str, horizon_key: str | None = None) -> float:
    """Hit rate for records where regime_label == regime_label_at_eval (stability filter).

    Only records with matching start/end regime are included. Records with a
    regime transition during the evaluation window are excluded by this filter.
    """
    records = load_records(agent_id, regime=regime, labeled_only=True, horizon_key=horizon_key)
    stable = [r for r in records if r.get("regime_label_at_eval") == r.get("regime_label")]
    return hit_rate(stable)


def baseline_naive(records: list[dict[str, Any]]) -> float:
    """Majority-class signal hit rate. Baseline: always predict the most common signal."""
    if not records:
        return 0.0
    signal_counts: Counter[str] = Counter(
        r["signal"] for r in records if r.get("outcome_direction_correct") is not None
    )
    if not signal_counts:
        return 0.0
    majority = signal_counts.most_common(1)[0][0]
    labeled = [r for r in records if r.get("outcome_direction_correct") is not None]
    return sum(1 for r in labeled if r["signal"] == majority and r["outcome_direction_correct"]) / len(labeled)


def baseline_rsi_rule(records: list[dict[str, Any]]) -> float:
    """RSI-rule baseline: RSI<30 -> buy (bullish correct), RSI>70 -> sell (bearish correct).

    Records without RSI data are excluded from this baseline.
    """
    eligible = [
        r
        for r in records
        if r.get("outcome_direction_correct") is not None and r.get("metadata", {}).get("rsi") is not None
    ]
    if not eligible:
        return 0.0
    correct = 0
    for r in eligible:
        rsi = r["metadata"]["rsi"]
        if rsi < 30 and r["signal"] == "bullish" and r["outcome_direction_correct"]:
            correct += 1
        elif rsi > 70 and r["signal"] == "bearish" and r["outcome_direction_correct"]:
            correct += 1
    return correct / len(eligible)
