"""Tests for src/utils/recommendation_log.py."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import aer.utils.recommendation_log as rec_log_module
from aer.utils.constants import REASONING_SUMMARY_MAX_CHARS, SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_record(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid recommendation record (no outcome keys)."""
    rec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rec_id": str(uuid.uuid4()),
        "agent_id": "test_agent",
        "ts_rec": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "ts_eval_open": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "ticker": "AAPL",
        "signal": "BUY",
        "horizons": [1, 5],
        "reasoning_summary": "Short summary.",
    }
    rec.update(overrides)
    return rec


def _patch_agents_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect _agents_root() and _reasoning_store_root() to tmp_path."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    monkeypatch.setattr(rec_log_module, "_agents_root", lambda: agents_dir)
    monkeypatch.setattr(rec_log_module, "_reasoning_store_root", lambda: agents_dir)
    return agents_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_write_rejects_naive_timestamp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ts_rec without +00:00 or Z suffix must raise ValueError."""
    _patch_agents_root(monkeypatch, tmp_path)
    record = _base_record(ts_rec="2026-05-17T10:00:00")  # naive — no UTC marker
    with pytest.raises(ValueError, match="ts_rec must be UTC ISO8601"):
        rec_log_module.write_recommendation(record)


def test_write_rejects_long_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """reasoning_summary exceeding REASONING_SUMMARY_MAX_CHARS must raise ValueError."""
    _patch_agents_root(monkeypatch, tmp_path)
    long_summary = "x" * (REASONING_SUMMARY_MAX_CHARS + 1)
    record = _base_record(reasoning_summary=long_summary)
    with pytest.raises(ValueError, match="reasoning_summary exceeds"):
        rec_log_module.write_recommendation(record)


def test_write_rejects_outcome_fields(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Any non-null outcome field present at write time must raise ValueError."""
    _patch_agents_root(monkeypatch, tmp_path)
    record = _base_record(outcome_direction_correct=1)
    with pytest.raises(ValueError, match="Outcome field.*must be null"):
        rec_log_module.write_recommendation(record)


def test_write_rejects_wrong_schema_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """schema_version != SCHEMA_VERSION must raise ValueError."""
    _patch_agents_root(monkeypatch, tmp_path)
    record = _base_record(schema_version=SCHEMA_VERSION - 1)
    with pytest.raises(ValueError, match="schema_version must be"):
        rec_log_module.write_recommendation(record)


def test_reasoning_full_ref_is_relative(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """reasoning_full_ref stored in the JSONL must be a relative path, not absolute."""
    agents_dir = _patch_agents_root(monkeypatch, tmp_path)

    # Redirect SQLite to a temp DB so sync_recommendation doesn't write to project DB.
    import aer.utils.db_sync as db_sync_module

    tmp_db = tmp_path / "test.db"

    def _fake_sync(record: dict[str, Any]) -> None:
        db_sync_module._DB_PATH = tmp_db  # type: ignore[attr-defined]
        db_sync_module._initialized = False
        db_sync_module.init_db()
        db_sync_module.sync_recommendation(record)

    monkeypatch.setattr(rec_log_module, "sync_recommendation", _fake_sync)

    rec_id = str(uuid.uuid4())
    record = _base_record(rec_id=rec_id, reasoning_full_text="Full reasoning text here.")

    rec_log_module.write_recommendation(record)

    # Read back from JSONL
    log_path = agents_dir / "test_agent" / "memory" / "recommendations.jsonl"
    assert log_path.exists(), "JSONL file was not created"
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "No records written"
    stored = json.loads(lines[-1])

    ref = stored.get("reasoning_full_ref", "")
    assert ref, "reasoning_full_ref is missing from stored record"
    assert not Path(ref).is_absolute(), f"reasoning_full_ref must be relative, got: {ref!r}"


def test_load_records_labeled_only_filters_null(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """load_records(labeled_only=True) must exclude records without a labeled outcome."""
    import aer.utils.db_sync as db_sync_module

    tmp_db = tmp_path / "recs.db"
    monkeypatch.setattr(db_sync_module, "_DB_PATH", tmp_db)
    monkeypatch.setattr(db_sync_module, "_initialized", False)
    db_sync_module.init_db()

    # Insert one recommendation with no outcome and one with a labeled outcome.
    now_iso = datetime.now(UTC).isoformat()

    rec_unlabeled: dict[str, Any] = {
        "rec_id": "rec-unlabeled",
        "agent_id": "agent_x",
        "agent_version": "abc",
        "ts_rec": now_iso,
        "ts_eval_open": now_iso,
        "ticker": "MSFT",
        "signal": "HOLD",
        "confidence": 0.6,
        "reasoning_summary": "No label",
        "reasoning_full_ref": None,
        "regime_label": None,
        "regime_context": None,
        "horizons": [5],
        "schema_version": SCHEMA_VERSION,
    }
    rec_labeled: dict[str, Any] = {**rec_unlabeled, "rec_id": "rec-labeled", "reasoning_summary": "Has label"}

    db_sync_module.sync_recommendation(rec_unlabeled)
    db_sync_module.sync_recommendation(rec_labeled)

    # Add outcome only for the labeled record
    outcome: dict[str, Any] = {
        "rec_id": "rec-labeled",
        "horizon_key": "5d",
        "horizon_days": 5,
        "outcome_ts": now_iso,
        "outcome_return_pct": 0.02,
        "outcome_direction_correct": 1,
        "regime_label_at_eval": None,
        "prices_used": None,
    }
    db_sync_module.sync_outcome(outcome, "agent_x")

    results = rec_log_module.load_records("agent_x", last_n_days=365, labeled_only=True, _db_path=tmp_db)
    rec_ids = {r["rec_id"] for r in results}

    assert "rec-labeled" in rec_ids, "Labeled record should be returned"
    assert "rec-unlabeled" not in rec_ids, "Unlabeled record must be excluded when labeled_only=True"


def test_horizons_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """horizons field validation: empty list, non-list, and negative int must raise ValueError."""
    _patch_agents_root(monkeypatch, tmp_path)

    # Empty list
    with pytest.raises(ValueError, match="horizons must be a non-empty list"):
        rec_log_module.write_recommendation(_base_record(horizons=[]))

    # Non-list
    with pytest.raises(ValueError, match="horizons must be a non-empty list"):
        rec_log_module.write_recommendation(_base_record(horizons=3))

    # Negative int
    with pytest.raises(ValueError, match="horizons must be a non-empty list"):
        rec_log_module.write_recommendation(_base_record(horizons=[-1]))

    # Valid — must not raise
    import aer.utils.db_sync as db_sync_module

    tmp_db = tmp_path / "hz.db"
    monkeypatch.setattr(db_sync_module, "_DB_PATH", tmp_db)
    monkeypatch.setattr(db_sync_module, "_initialized", False)
    db_sync_module.init_db()

    def _fake_sync(record: dict[str, Any]) -> None:
        db_sync_module.sync_recommendation(record)

    monkeypatch.setattr(rec_log_module, "sync_recommendation", _fake_sync)

    rec_log_module.write_recommendation(_base_record(horizons=[1, 3]))  # must not raise
