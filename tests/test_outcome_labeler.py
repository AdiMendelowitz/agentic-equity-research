"""Tests for src/utils/outcome_labeler.py.

All file I/O is redirected to tmp_path. now_utc() is monkeypatched to control
window closure. price_fetcher is a simple injected callable.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import aer.utils.outcome_labeler as labeler_module
from aer.utils.constants import OUTCOME_RETURN_PLAUSIBILITY_CAP, SCHEMA_VERSION
from aer.utils.outcome_labeler import label_pending_outcomes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REC_TS = datetime(2026, 1, 1, 9, 0, 0, tzinfo=UTC)
_EVAL_OPEN = datetime(2026, 1, 1, 9, 30, 0, tzinfo=UTC)  # 30 min after rec
_HORIZON = 1  # D+1
_HORIZON_KEY = "D+1"
_ENTRY_PRICE = 100.0
_EXIT_PRICE = 105.0  # +5% — bullish correct, within cap


def _ts(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _base_record(**overrides: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rec_id": str(uuid.uuid4()),
        "agent_id": "agent_alpha",
        "ts_rec": _ts(_REC_TS),
        "ts_eval_open": _ts(_EVAL_OPEN),
        "ticker": "AAPL",
        "signal": "bullish",
        "horizons": [_HORIZON],
        "regime_label": "trending",
        "reasoning_summary": "test",
    }
    rec.update(overrides)
    return rec


def _write_rec(agents_root: Path, record: dict[str, Any]) -> None:
    agent_dir = agents_root / record["agent_id"] / "memory"
    agent_dir.mkdir(parents=True, exist_ok=True)
    rec_file = agent_dir / "recommendations.jsonl"
    with rec_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _read_outcomes(agents_root: Path, agent_id: str) -> list[dict[str, Any]]:
    path = agents_root / agent_id / "memory" / "outcomes.jsonl"
    if not path.exists():
        return []
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def _patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, now_dt: datetime) -> Path:
    """Redirect agents_root and now_utc; disable sync_outcome (no DB in tests)."""
    agents_root = tmp_path / "agents"
    agents_root.mkdir()
    monkeypatch.setattr(labeler_module, "_agents_root", lambda: agents_root)
    monkeypatch.setattr(labeler_module, "now_utc", lambda: now_dt)
    monkeypatch.setattr(labeler_module, "sync_outcome", lambda outcome, agent_id: None)
    return agents_root


# D+1 window closes at ts_eval_open + 1 day; set now to safely past that
_NOW_PAST = _EVAL_OPEN + timedelta(days=2)


def _simple_fetcher(ticker: str, date_str: str) -> float:
    entry_date = _EVAL_OPEN.strftime("%Y-%m-%d")
    if date_str == entry_date:
        return _ENTRY_PRICE
    return _EXIT_PRICE


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_lookahead_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ts_rec < ts_eval_open — record processes without LookaheadViolationError."""
    agents_root = _patch(monkeypatch, tmp_path, _NOW_PAST)
    rec = _base_record()
    _write_rec(agents_root, rec)

    label_pending_outcomes(_simple_fetcher)

    outcomes = _read_outcomes(agents_root, rec["agent_id"])
    assert len(outcomes) == 1
    assert outcomes[0]["rec_id"] == rec["rec_id"]


def test_lookahead_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """ts_rec >= ts_eval_open — record is skipped (LookaheadViolationError logged, no abort)."""
    agents_root = _patch(monkeypatch, tmp_path, _NOW_PAST)
    # Make ts_rec equal to ts_eval_open (violation)
    rec = _base_record(ts_rec=_ts(_EVAL_OPEN), ts_eval_open=_ts(_EVAL_OPEN))
    _write_rec(agents_root, rec)

    # Must not raise — violation is logged and skipped
    label_pending_outcomes(_simple_fetcher)

    outcomes = _read_outcomes(agents_root, rec["agent_id"])
    assert len(outcomes) == 0


def test_original_record_not_mutated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """recommendations.jsonl must be unchanged after labeling."""
    agents_root = _patch(monkeypatch, tmp_path, _NOW_PAST)
    rec = _base_record()
    _write_rec(agents_root, rec)
    rec_file = agents_root / rec["agent_id"] / "memory" / "recommendations.jsonl"
    original_content = rec_file.read_text(encoding="utf-8")

    label_pending_outcomes(_simple_fetcher)

    assert rec_file.read_text(encoding="utf-8") == original_content


def test_outcome_written_to_outcomes_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Outcome appears in outcomes.jsonl, not in recommendations.jsonl."""
    agents_root = _patch(monkeypatch, tmp_path, _NOW_PAST)
    rec = _base_record()
    _write_rec(agents_root, rec)

    label_pending_outcomes(_simple_fetcher)

    # outcomes.jsonl must have the record
    outcomes = _read_outcomes(agents_root, rec["agent_id"])
    assert len(outcomes) == 1
    assert outcomes[0]["horizon_key"] == _HORIZON_KEY

    # recommendations.jsonl must NOT contain outcome fields with real values
    rec_file = agents_root / rec["agent_id"] / "memory" / "recommendations.jsonl"
    raw = json.loads(rec_file.read_text(encoding="utf-8").strip())
    assert "outcome_return_pct" not in raw or raw.get("outcome_return_pct") is None


def test_idempotent_on_rerun(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Calling label_pending_outcomes twice must not produce duplicate outcome records."""
    agents_root = _patch(monkeypatch, tmp_path, _NOW_PAST)
    rec = _base_record()
    _write_rec(agents_root, rec)

    label_pending_outcomes(_simple_fetcher)
    label_pending_outcomes(_simple_fetcher)

    outcomes = _read_outcomes(agents_root, rec["agent_id"])
    assert len(outcomes) == 1


def test_implausible_return_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """|return| >= OUTCOME_RETURN_PLAUSIBILITY_CAP — record must be skipped entirely."""
    agents_root = _patch(monkeypatch, tmp_path, _NOW_PAST)
    rec = _base_record()
    _write_rec(agents_root, rec)

    cap = OUTCOME_RETURN_PLAUSIBILITY_CAP
    implausible_exit = _ENTRY_PRICE * (1 + cap + 0.01)  # just above cap

    def bad_fetcher(ticker: str, date_str: str) -> float:
        entry_date = _EVAL_OPEN.strftime("%Y-%m-%d")
        return _ENTRY_PRICE if date_str == entry_date else implausible_exit

    label_pending_outcomes(bad_fetcher)

    outcomes = _read_outcomes(agents_root, rec["agent_id"])
    assert len(outcomes) == 0


def test_window_not_closed_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Horizon window has not elapsed — record must be skipped."""
    # now is only 12 hours after eval_open, but D+1 requires 24+ hours
    now_early = _EVAL_OPEN + timedelta(hours=12)
    agents_root = _patch(monkeypatch, tmp_path, now_early)
    rec = _base_record()
    _write_rec(agents_root, rec)

    label_pending_outcomes(_simple_fetcher)

    outcomes = _read_outcomes(agents_root, rec["agent_id"])
    assert len(outcomes) == 0


def test_delisted_ticker_produces_null_outcome(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """price_fetcher raises → null outcome written (outcome_return_pct=None, direction=None)."""
    agents_root = _patch(monkeypatch, tmp_path, _NOW_PAST)
    rec = _base_record(ticker="DEAD")
    _write_rec(agents_root, rec)

    def failing_fetcher(ticker: str, date_str: str) -> float:
        raise ValueError(f"No price data for delisted ticker {ticker}")

    label_pending_outcomes(failing_fetcher)

    outcomes = _read_outcomes(agents_root, rec["agent_id"])
    assert len(outcomes) == 1
    assert outcomes[0]["outcome_return_pct"] is None
    assert outcomes[0]["outcome_direction_correct"] is None


def test_regime_label_at_eval_stored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """outcome record contains regime_label_at_eval copied from the recommendation record."""
    agents_root = _patch(monkeypatch, tmp_path, _NOW_PAST)
    rec = _base_record(regime_label="volatile")
    _write_rec(agents_root, rec)

    label_pending_outcomes(_simple_fetcher)

    outcomes = _read_outcomes(agents_root, rec["agent_id"])
    assert len(outcomes) == 1
    assert outcomes[0]["regime_label_at_eval"] == "volatile"
