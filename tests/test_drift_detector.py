"""Tests for CUSUMDetector — all writes go to tmp_path, never to real project dirs."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aer.utils.constants import CUSUM_THRESHOLD
from aer.utils.drift_detector import CUSUMDetector


def _make_detector(tmp_path: Path, agent_id: str = "test_agent") -> CUSUMDetector:
    return CUSUMDetector(agent_id=agent_id, agents_root=tmp_path)


def _state_path(tmp_path: Path, agent_id: str = "test_agent") -> Path:
    return tmp_path / agent_id / "memory" / "cusum_state.json"


def test_no_alert_below_threshold(tmp_path: Path) -> None:
    det = _make_detector(tmp_path)
    # Single update with perfect hit rate: deviation = 0.50 - 1.0 - 0.02 = -0.52 → clamped to 0
    result = det.update(hit_rate=1.0)
    assert result is False
    assert det._cusum < CUSUM_THRESHOLD


def test_alert_above_threshold(tmp_path: Path) -> None:
    det = _make_detector(tmp_path)
    # Each update with hit_rate=0.0: deviation = 0.50 - 0.0 - 0.02 = 0.48 per call
    # Need cusum >= 5.0 → ceil(5.0 / 0.48) = 11 calls
    alert_fired = False
    with patch.object(det, "_log_drift_event"):
        for _ in range(15):
            if det.update(hit_rate=0.0):
                alert_fired = True
                break
    assert alert_fired is True
    assert det._cusum >= CUSUM_THRESHOLD


def test_state_persisted_after_update(tmp_path: Path) -> None:
    det = _make_detector(tmp_path)
    state_file = _state_path(tmp_path)

    assert not state_file.exists()

    with patch.object(det, "_log_drift_event"):
        det.update(hit_rate=0.0)

    assert state_file.exists()
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert "cusum" in data
    assert data["agent_id"] == "test_agent"
    assert data["cusum"] > 0.0


def test_state_reloaded_on_init(tmp_path: Path) -> None:
    agent_id = "reload_agent"
    det1 = _make_detector(tmp_path, agent_id=agent_id)

    with patch.object(det1, "_log_drift_event"):
        for _ in range(5):
            det1.update(hit_rate=0.0)

    cusum_after = det1._cusum

    det2 = _make_detector(tmp_path, agent_id=agent_id)
    assert det2._cusum == pytest.approx(cusum_after)


def test_reset_zeroes_state(tmp_path: Path) -> None:
    det = _make_detector(tmp_path)

    with patch.object(det, "_log_drift_event"):
        for _ in range(5):
            det.update(hit_rate=0.0)

    assert det._cusum > 0.0

    det.reset()

    assert det._cusum == 0.0

    state = json.loads(_state_path(tmp_path).read_text(encoding="utf-8"))
    assert state["cusum"] == 0.0

    det2 = _make_detector(tmp_path)
    assert det2._cusum == 0.0
