"""Tests for src/utils/conflict_classifier.py."""

from unittest.mock import patch

from aer.utils.conflict_classifier import _conflicts_exceed_threshold, classify_conflict
from aer.utils.constants import CONF_HIGH_CONVICTION_HIGH_RISK, CONF_MACRO_OVERRIDE


def _view(
    agent: str,
    view: str = "BUY",
    confidence: float = 0.8,
    regime_label: str = "RISK_ON",
) -> dict:
    """Minimal view dict for classifier tests — matches ProfessorAgentView field names."""
    return {
        "agent": agent,
        "view": view,
        "confidence": confidence,
        "regime_label": regime_label,
        "ticker": "AAPL",
        "rationale": {},
        "risks": [],
        "ts": "2026-01-01T00:00:00+00:00",
    }


# --- _conflicts_exceed_threshold ---


def test_two_conflict_max_severity_aggregation() -> None:
    """Two severity-5 conflicts do NOT clear threshold-6 — max gate, not additive."""
    conflicts = [{"severity": 5, "pattern": "A"}, {"severity": 5, "pattern": "B"}]
    assert _conflicts_exceed_threshold(conflicts) is False


def test_single_severity_6_exceeds_threshold() -> None:
    """Single severity-6 conflict clears the threshold."""
    conflicts = [{"severity": 6, "pattern": "X"}]
    assert _conflicts_exceed_threshold(conflicts) is True


# --- classify_conflict ---


def test_classify_conflict_empty_views_returns_empty() -> None:
    """Empty view list returns empty conflict list."""
    assert classify_conflict([], {}, {}) == []


def test_classify_conflict_sorts_descending() -> None:
    """classify_conflict result is sorted by severity descending."""
    # Patch two check functions to return controlled dicts at severity 7 and 6.
    # We patch _check_high_conviction_high_risk (normally severity 9) and
    # _check_conviction_split (normally severity 6) to inject controlled conflicts.
    sev7_conflict = {"pattern": "FAKE_HIGH", "severity": 7, "views": [], "details": {}}
    sev6_conflict = {"pattern": "FAKE_LOW", "severity": 6, "views": [], "details": {}}

    with (
        patch("aer.utils.conflict_classifier._check_high_conviction_high_risk", return_value=[sev7_conflict]),
        patch("aer.utils.conflict_classifier._check_macro_override", return_value=[sev6_conflict]),
        patch("aer.utils.conflict_classifier._check_smart_money_flow", return_value=[]),
        patch("aer.utils.conflict_classifier._check_sentiment_divergence", return_value=[]),
        patch("aer.utils.conflict_classifier._check_technical_reversal", return_value=[]),
        patch("aer.utils.conflict_classifier._check_regime_mismatch", return_value=[]),
        patch("aer.utils.conflict_classifier._check_conviction_split", return_value=[]),
        patch("aer.utils.conflict_classifier._check_ml_signal_mismatch", return_value=[]),
    ):
        result = classify_conflict([], {}, {})

    assert len(result) == 2
    assert result[0]["severity"] == 7
    assert result[1]["severity"] == 6


def test_macro_override_detected() -> None:
    """Macro SELL + 2 BUY agents at confidence 0.7+ returns CONF_MACRO_OVERRIDE conflict."""
    views = [
        _view("macro", view="SELL", confidence=0.9),
        _view("fundamental", view="BUY", confidence=0.75),
        _view("technical", view="BUY", confidence=0.7),
    ]
    result = classify_conflict(views, {}, {})
    patterns = [c["pattern"] for c in result]
    assert CONF_MACRO_OVERRIDE in patterns


def test_high_conviction_high_risk_detected() -> None:
    """3 BUY agents at confidence 0.8 + risk score 8 returns CONF_HIGH_CONVICTION_HIGH_RISK."""
    views = [
        _view("fundamental", view="BUY", confidence=0.8),
        _view("technical", view="BUY", confidence=0.8),
        _view("sentiment", view="BUY", confidence=0.8),
    ]
    result = classify_conflict(views, {}, {"score": 8})
    patterns = [c["pattern"] for c in result]
    assert CONF_HIGH_CONVICTION_HIGH_RISK in patterns


def test_sub_threshold_conflicts_not_returned() -> None:
    """ml_signal_mismatch (severity 5) with ml_tension=True in macro returns empty list."""
    views = [_view("fundamental", view="BUY", confidence=0.8)]
    # ml_tension triggers _check_ml_signal_mismatch at severity 5 (below threshold 6)
    result = classify_conflict(views, {"ml_tension": True}, {})
    # No conflict should be above threshold from ml_tension alone
    assert all(c["severity"] >= 6 for c in result)
    # With only one BUY agent and no other patterns triggering, result should be empty
    assert result == []


def test_conviction_split_detected() -> None:
    """1 BUY at confidence 0.8 and 1 SELL at confidence 0.8 returns CONF_CONVICTION_SPLIT conflict."""
    views = [
        _view("fundamental", view="BUY", confidence=0.8),
        _view("technical", view="SELL", confidence=0.8),
    ]
    result = classify_conflict(views, {}, {})
    patterns = [c["pattern"] for c in result]
    assert any("CONVICTION_SPLIT" in p for p in patterns)
