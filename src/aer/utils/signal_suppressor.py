"""Signal suppression rule — pure Python, no LLM.

CUSUM hysteresis note: CUSUM fires at effective threshold ~0.50 (early warning).
Suppression fires at 0.45 (backstop). The 5-point gap is intentional — CUSUM
detects sustained drift before suppression activates.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import portalocker

from aer.utils.constants import (
    SUPPRESSION_HIT_RATE_FLOOR,
    SUPPRESSION_MIN_SAMPLES,
    SUPPRESSION_REVIEW_WINDOW_DAYS,
)
from aer.utils.recommendation_log import load_records

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_logger = logging.getLogger(__name__)


def should_suppress(
    agent_id: str,
    signal: str,
    regime: str,
    *,
    _log_path: Path | None = None,
    _db_path: Path | None = None,
) -> bool:
    """Return True if the agent's signal should be suppressed in this regime.

    Uses labeled_only=True — never operates on null outcomes.
    Suppressed signals are logged and flagged for human review, never silently dropped.
    """
    records = load_records(
        agent_id,
        regime=regime,
        last_n_days=SUPPRESSION_REVIEW_WINDOW_DAYS,
        labeled_only=True,
        signal=signal,
        _db_path=_db_path,
    )
    if len(records) < SUPPRESSION_MIN_SAMPLES:
        return False  # insufficient data — do not suppress
    correct = sum(1 for r in records if r.get("outcome_direction_correct"))
    rate = correct / len(records)
    if rate < SUPPRESSION_HIT_RATE_FLOOR:
        _log_suppression(agent_id, signal, regime, rate, len(records), _log_path)
        return True
    return False


def _log_suppression(
    agent_id: str,
    signal: str,
    regime: str,
    hit_rate: float,
    sample_count: int,
    log_path: Path | None,
) -> None:
    path = log_path or (_PROJECT_ROOT / "docs" / "suppression_log.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": datetime.now(UTC).isoformat(),
        "agent_id": agent_id,
        "signal": signal,
        "regime": regime,
        "hit_rate": round(hit_rate, 4),
        "sample_count": sample_count,
    }
    with portalocker.Lock(str(path), mode="a", flags=portalocker.LOCK_EX, encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")
    _logger.warning(
        "Signal suppressed: agent=%s signal=%s regime=%s hit_rate=%.3f n=%d",
        agent_id,
        signal,
        regime,
        hit_rate,
        sample_count,
    )
