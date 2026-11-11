"""CUSUM drift detector per agent. State persisted to cusum_state.json on every update.

In-memory-only state is prohibited — it resets on every deploy and makes
slow-building drift undetectable.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from aer.utils.constants import (
    CUSUM_SLACK,
    CUSUM_TARGET_HIT_RATE,
    CUSUM_THRESHOLD,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_logger = logging.getLogger(__name__)


class CUSUMDetector:
    """Page-Hinkley CUSUM detector for one agent's rolling hit rate.

    Alert fires when cumulative sum of deviations below target exceeds CUSUM_THRESHOLD.
    State is persisted to agents/{agent_id}/memory/cusum_state.json after every update.
    """

    def __init__(self, agent_id: str, agents_root: Path | None = None) -> None:
        self._agent_id = agent_id
        self._root = agents_root or (_PROJECT_ROOT / "agents")
        self._state_path = self._root / agent_id / "memory" / "cusum_state.json"
        self._cusum: float = 0.0
        self._load()

    def _load(self) -> None:
        if self._state_path.exists():
            try:
                state = json.loads(self._state_path.read_text(encoding="utf-8"))
                self._cusum = float(state.get("cusum", 0.0))
            except (json.JSONDecodeError, KeyError, ValueError):
                self._cusum = 0.0

    def _save(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps({"cusum": self._cusum, "agent_id": self._agent_id}),
            encoding="utf-8",
        )

    def update(self, hit_rate: float) -> bool:
        """Update CUSUM with latest hit rate. Returns True if drift threshold crossed."""
        deviation = CUSUM_TARGET_HIT_RATE - hit_rate - CUSUM_SLACK
        self._cusum = max(0.0, self._cusum + deviation)
        self._save()

        if self._cusum >= CUSUM_THRESHOLD:
            self._log_drift_event()
            return True
        return False

    def reset(self) -> None:
        """Reset accumulator to zero and persist."""
        self._cusum = 0.0
        self._save()

    def _log_drift_event(self) -> None:
        drift_log = _PROJECT_ROOT / "docs" / "drift_events.jsonl"
        drift_log.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": datetime.now(UTC).isoformat(),
            "agent_id": self._agent_id,
            "drift_type": "agent",
            "cusum": self._cusum,
        }
        with open(drift_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        _logger.warning("CUSUM drift alert: agent=%s cusum=%.3f", self._agent_id, self._cusum)
