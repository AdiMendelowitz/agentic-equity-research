"""Tests for signal_suppressor — all writes go to tmp_path, never to real project dirs."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aer.utils.constants import SUPPRESSION_MIN_SAMPLES
from aer.utils.signal_suppressor import should_suppress


def _make_records(n: int, correct: int) -> list[dict]:
    """Return n records where `correct` of them have outcome_direction_correct=True."""
    records = []
    for i in range(n):
        records.append({"outcome_direction_correct": i < correct})
    return records


def test_insufficient_samples_no_suppress(tmp_path: Path) -> None:
    """Fewer than SUPPRESSION_MIN_SAMPLES labeled records → always False."""
    few_records = _make_records(SUPPRESSION_MIN_SAMPLES - 1, 0)  # all wrong, still not enough samples

    with patch("aer.utils.signal_suppressor.load_records", return_value=few_records):
        result = should_suppress(
            "agent_x",
            "BUY",
            "trending",
            _log_path=tmp_path / "suppression_log.jsonl",
        )

    assert result is False


def test_low_hit_rate_suppresses(tmp_path: Path) -> None:
    """hit_rate < SUPPRESSION_HIT_RATE_FLOOR with >= SUPPRESSION_MIN_SAMPLES records → True."""
    n = SUPPRESSION_MIN_SAMPLES
    # 8/20 = 0.40 < 0.45
    records = _make_records(n, correct=8)
    log_path = tmp_path / "suppression_log.jsonl"

    with patch("aer.utils.signal_suppressor.load_records", return_value=records):
        result = should_suppress(
            "agent_x",
            "BUY",
            "trending",
            _log_path=log_path,
        )

    assert result is True


def test_adequate_hit_rate_passes(tmp_path: Path) -> None:
    """hit_rate >= SUPPRESSION_HIT_RATE_FLOOR with >= SUPPRESSION_MIN_SAMPLES records → False."""
    n = SUPPRESSION_MIN_SAMPLES
    # 9/20 = 0.45 — exactly at floor, should pass (not suppressed)
    records = _make_records(n, correct=9)

    with patch("aer.utils.signal_suppressor.load_records", return_value=records):
        result = should_suppress(
            "agent_x",
            "BUY",
            "trending",
            _log_path=tmp_path / "suppression_log.jsonl",
        )

    assert result is False


def test_never_sums_over_null_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_records must be called with labeled_only=True — null outcomes never aggregated."""
    captured: dict = {}

    def fake_load_records(agent_id, *, regime=None, last_n_days=30, labeled_only=True, signal=None, _db_path=None):
        captured["labeled_only"] = labeled_only
        return []  # 0 records → no suppress

    monkeypatch.setattr("aer.utils.signal_suppressor.load_records", fake_load_records)

    should_suppress("agent_x", "BUY", "trending")

    assert captured.get("labeled_only") is True, "load_records must be called with labeled_only=True"


def test_suppression_logged_on_suppress(tmp_path: Path) -> None:
    """When suppression fires, a valid JSON entry is appended to the log file."""
    n = SUPPRESSION_MIN_SAMPLES
    # 0/20 = 0.0 — clearly below floor
    records = _make_records(n, correct=0)
    log_path = tmp_path / "suppression_log.jsonl"

    with patch("aer.utils.signal_suppressor.load_records", return_value=records):
        result = should_suppress(
            "agent_alpha",
            "SELL",
            "volatile",
            _log_path=log_path,
        )

    assert result is True
    assert log_path.exists(), "suppression_log.jsonl was not created"

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, f"Expected 1 log entry, got {len(lines)}"

    entry = json.loads(lines[0])
    assert entry["agent_id"] == "agent_alpha"
    assert entry["signal"] == "SELL"
    assert entry["regime"] == "volatile"
    assert entry["hit_rate"] == 0.0
    assert entry["sample_count"] == n
    assert "ts" in entry
