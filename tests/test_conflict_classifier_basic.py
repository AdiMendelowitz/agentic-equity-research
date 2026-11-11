"""TDD RED: basic failing tests for src/utils/conflict_classifier.py."""

from aer.utils.conflict_classifier import _conflicts_exceed_threshold, classify_conflict


def test_conflicts_exceed_threshold_max_gate_not_additive() -> None:
    """Two severity-5 conflicts do NOT clear threshold-6 — not additive."""
    conflicts = [{"severity": 5, "pattern": "A"}, {"severity": 5, "pattern": "B"}]
    assert _conflicts_exceed_threshold(conflicts) is False


def test_conflicts_exceed_threshold_single_at_threshold() -> None:
    """Single severity-6 conflict clears the threshold."""
    conflicts = [{"severity": 6, "pattern": "X"}]
    assert _conflicts_exceed_threshold(conflicts) is True


def test_classify_conflict_empty_views_returns_empty() -> None:
    """Empty view list returns empty conflict list."""
    assert classify_conflict([], {}, {}) == []


def test_classify_conflict_returns_only_above_threshold() -> None:
    """classify_conflict filters out sub-threshold conflicts."""
    result = classify_conflict([], {"ml_tension": True}, {})
    assert all(c["severity"] >= 6 for c in result)
