"""Nightly outcome labeler — appends labeled outcome records to outcomes.jsonl.

Reads recommendation records (labeled_only=False), checks window closure per horizon,
fetches closing price, computes return, guards against lookahead and implausible returns,
then appends one outcome record per (rec_id, horizon_key) pair. Fully idempotent.
"""

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import portalocker

from aer.utils.constants import OUTCOME_RETURN_PLAUSIBILITY_CAP, SCHEMA_VERSION
from aer.utils.db_sync import sync_outcome

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LookaheadViolationError(Exception):
    """Raised when ts_rec >= ts_eval_open — recommendation recorded after evaluation open."""


def now_utc() -> datetime:
    """Return current UTC datetime. Never use datetime.utcnow()."""
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _agents_root() -> Path:
    return _PROJECT_ROOT / "agents"


def _load_existing_outcomes(agent_id: str, agents_root: Path) -> set[tuple[str, str]]:
    """Return set of (rec_id, horizon_key) pairs already written to outcomes.jsonl."""
    path = agents_root / agent_id / "memory" / "outcomes.jsonl"
    if not path.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rec = json.loads(line)
                seen.add((rec["rec_id"], rec["horizon_key"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return seen


def _parse_utc(ts: str) -> datetime:
    """Parse UTC ISO8601 string (with Z or +00:00) to aware datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _direction_correct(signal: str, return_pct: float) -> bool:
    if signal == "bullish":
        return return_pct > 0
    if signal == "bearish":
        return return_pct < 0
    return True  # neutral — always correct


def _append_outcome(agent_id: str, agents_root: Path, outcome: dict) -> None:
    """Append one outcome record to outcomes.jsonl under exclusive file lock."""
    out_path = agents_root / agent_id / "memory" / "outcomes.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with portalocker.Lock(str(out_path), mode="a", flags=portalocker.LOCK_EX, encoding="utf-8") as fh:
        fh.write(json.dumps(outcome) + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def label_pending_outcomes(price_fetcher: Callable[[str, str], float]) -> None:
    """Label all unlabeled (rec_id, horizon_key) pairs across all agents.

    price_fetcher(ticker, date_str) -> float  — returns split-adjusted close for date_str.
    Raises any exception on fetch failure (delisted/unavailable) → null outcome written.
    """
    agents_root = _agents_root()
    if not agents_root.exists():
        logger.warning("agents/ directory does not exist — nothing to label")
        return

    for agent_dir in sorted(agents_root.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_id = agent_dir.name
        rec_file = agent_dir / "memory" / "recommendations.jsonl"
        if not rec_file.exists():
            continue

        # Load idempotency set once per agent before processing any records
        existing = _load_existing_outcomes(agent_id, agents_root)

        for line in rec_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("Bad JSON in %s: %s", rec_file, exc)
                continue

            rec_id: str = record.get("rec_id", "")
            ticker: str = record.get("ticker", "")
            signal: str = record.get("signal", "neutral")
            horizons: list[int] = record.get("horizons", [])
            ts_rec_str: str = record.get("ts_rec", "")
            ts_eval_open_str: str = record.get("ts_eval_open", "")
            regime_label: str = record.get("regime_label", "ranging")

            try:
                ts_rec_dt = _parse_utc(ts_rec_str)
                ts_eval_open_dt = _parse_utc(ts_eval_open_str)
            except (ValueError, TypeError) as exc:
                logger.error("Cannot parse timestamps for rec_id=%s: %s", rec_id, exc)
                continue

            # Lookahead guard — recommendation must precede evaluation window open
            if ts_rec_dt >= ts_eval_open_dt:
                err = LookaheadViolationError(
                    f"rec_id={rec_id}: ts_rec={ts_rec_str} >= ts_eval_open={ts_eval_open_str}"
                )
                logger.error("LOOKAHEAD VIOLATION — skipping rec_id=%s: %s", rec_id, err)
                continue

            for horizon_days in horizons:
                horizon_key = f"D+{horizon_days}"

                # Idempotency — skip if already labeled
                if (rec_id, horizon_key) in existing:
                    continue

                # Window closure check — only label once the horizon has elapsed
                eval_close_dt = ts_eval_open_dt + timedelta(days=horizon_days)
                if now_utc() < eval_close_dt:
                    continue

                # Entry price: closing price on ts_eval_open date
                entry_date_str = ts_eval_open_dt.strftime("%Y-%m-%d")
                exit_date_str = eval_close_dt.strftime("%Y-%m-%d")
                outcome_ts = now_utc().isoformat()

                outcome_return_pct: float | None = None
                outcome_direction_correct: bool | None = None
                entry_price: float | None = None
                exit_price: float | None = None

                try:
                    entry_price = price_fetcher(ticker, entry_date_str)
                    exit_price = price_fetcher(ticker, exit_date_str)

                    raw_return = (exit_price - entry_price) / entry_price

                    # Plausibility cap — returns > 30% on short horizons are suspect
                    if abs(raw_return) >= OUTCOME_RETURN_PLAUSIBILITY_CAP:
                        logger.warning(
                            "Implausible return %.4f for rec_id=%s horizon=%s — skipping",
                            raw_return,
                            rec_id,
                            horizon_key,
                        )
                        continue

                    outcome_return_pct = raw_return
                    outcome_direction_correct = _direction_correct(signal, raw_return)

                except Exception as exc:
                    # Delisted or fetch failed — write null outcome so we don't retry forever
                    logger.warning(
                        "Price fetch failed for %s on %s/%s (rec_id=%s): %s",
                        ticker,
                        entry_date_str,
                        exit_date_str,
                        rec_id,
                        exc,
                    )
                    outcome_return_pct = None
                    outcome_direction_correct = None

                outcome: dict = {
                    "rec_id": rec_id,
                    "horizon_key": horizon_key,
                    "horizon_days": horizon_days,
                    "agent_id": agent_id,
                    "ticker": ticker,
                    "outcome_ts": outcome_ts,
                    "outcome_return_pct": outcome_return_pct,
                    "outcome_direction_correct": outcome_direction_correct,
                    # regime at eval — copy from record; live ATR/ADX not available in labeler
                    "regime_label_at_eval": regime_label,
                    "prices_used": {
                        "entry_date": entry_date_str,
                        "exit_date": exit_date_str,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                    },
                    "schema_version": SCHEMA_VERSION,
                }

                _append_outcome(agent_id, agents_root, outcome)
                # Update idempotency set in memory to guard against duplicates within same run
                existing.add((rec_id, horizon_key))
                sync_outcome(outcome, agent_id)
