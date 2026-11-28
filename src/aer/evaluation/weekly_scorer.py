"""Evaluation DB - write, resolve, and score investment brief outcomes.

Thread safety: _DB_LOCK (threading.Lock) for in-process serialisation.
Cross-process: PRAGMA busy_timeout=5000 on each connection.

Note on synth_correct:
synth_correct compares a weeks-to-months LLM thesis against 5-day price movement.
This metric is logged for observability. It is NOT used in should_trigger_review()
to make model replacement decisions - the five-day proxy does not
match the horizon of the LLM thesis, and which side the mismatch favours is not known.

What the stored columns mean:
- actual_direction is UP when the five-day return on adjusted closes is above zero and DOWN otherwise. There is
  no cost band, so a wrong BUY fixes the outcome as DOWN and a wrong SELL or AVOID fixes it as UP.
- ml_direction is the direction of the member the gate checked (aer.ml.models.selected_prediction). Rows written
  before 17 Nov 2026, which include every published aggregate, carry the LightGBM member's direction once the
  two-model ensemble was added and the selected model's direction before it. The ensemble direction is not stored,
  and ml_correct scores the stored direction.
- The ML label in aer.ml.features is a return above a 10 basis point cost on unadjusted closes, so ml_correct
  and the training target are not the same quantity.
- v0.3 experiment results never go into this store.
"""

import json
import logging
import sqlite3
import threading
import time
import uuid
from datetime import date

import yfinance as yf

from aer.evaluation.eval_models import EVAL_SCHEMA, EVAL_UNIQUE_INDEX, compute_synth_correct
from aer.evaluation.spec import DEFAULT_SPEC
from aer.ml.constants import (
    BASELINE_WINDOW_WEEKS,
    DRIFT_STD_MULTIPLIER,
    MAX_RESOLUTION_ATTEMPTS,
    MIN_OOS_FOR_DECISIONS,
    MIN_OOS_FOR_MEANINGFUL_ACCURACY,
)
from aer.models import InvestmentBrief
from aer.paths import PROJECT_ROOT as _PROJECT_ROOT
from aer.utils.db import connect_db
from aer.utils.trading_calendar import add_trading_days

logger = logging.getLogger(__name__)


_EVAL_DB_PATH = _PROJECT_ROOT / "eval_log" / "eval.db"
_DB_LOCK = threading.Lock()
_RUN_ID = uuid.uuid4().hex
_skipped_reruns = 0


class DuplicateTickerDayError(RuntimeError):
    """The store holds more than one row for a ticker-day, so the unique index cannot be created."""


def _connect() -> sqlite3.Connection:
    _EVAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return connect_db(str(_EVAL_DB_PATH), busy_timeout_ms=5000, row_factory=sqlite3.Row)


def _has_table(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='agent_eval'").fetchone() is not None


def _init_db() -> None:
    with _DB_LOCK:
        conn = _connect()
        try:
            if _has_table(conn):
                dupes = conn.execute(
                    "SELECT COUNT(*) FROM (SELECT 1 FROM agent_eval GROUP BY ticker, brief_date HAVING COUNT(*) > 1)"
                ).fetchone()[0]
                if dupes:
                    raise DuplicateTickerDayError(
                        f"{_EVAL_DB_PATH} holds {dupes} ticker-day{'' if dupes == 1 else 's'} with more than one row. "
                        "Nothing was changed. "
                        "Back up the file, then call remove_duplicate_ticker_days() once to keep the lowest id per day."
                    )
            for stmt in EVAL_SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            if "run_id" not in {r[1] for r in conn.execute("PRAGMA table_info(agent_eval)")}:
                try:
                    conn.execute("ALTER TABLE agent_eval ADD COLUMN run_id TEXT")
                except sqlite3.OperationalError:
                    # Another process may have added the column first, so re-raise only if it is still missing.
                    if "run_id" not in {r[1] for r in conn.execute("PRAGMA table_info(agent_eval)")}:
                        raise
            conn.execute(EVAL_UNIQUE_INDEX)
            conn.commit()
        finally:
            conn.close()


def remove_duplicate_ticker_days() -> int:
    """Delete every agent_eval row that is not the lowest id of its ticker-day. Returns the number deleted.

    Explicit and never called by _init_db. It does not create the store or its table. It deletes permanently from the
    store at _EVAL_DB_PATH, so copy the file first.
    """
    if not _EVAL_DB_PATH.is_file():
        return 0
    with _DB_LOCK:
        conn = connect_db(str(_EVAL_DB_PATH), busy_timeout_ms=5000)
        try:
            if not _has_table(conn):
                return 0
            deleted = conn.execute(
                "DELETE FROM agent_eval WHERE id NOT IN (SELECT MIN(id) FROM agent_eval GROUP BY ticker, brief_date)"
            ).rowcount
            conn.commit()
            return deleted
        finally:
            conn.close()


def record_brief_generated(brief: InvestmentBrief, quality_metrics: dict) -> None:
    """Write a PENDING eval row for this brief. Call after invoke_pipeline() returns."""
    global _skipped_reruns
    _init_db()
    today_str = date.today().isoformat()
    resolution_date = add_trading_days(date.today(), DEFAULT_SPEC.resolution_lag_trading_days).isoformat()
    ml = brief.ml_prediction
    now = time.time()

    with _DB_LOCK:
        conn = _connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO agent_eval (
                    ticker, brief_date, recommendation, conviction,
                    ml_direction, ml_prob_up_raw, ml_cv_f1_macro, ml_cv_f1_std,
                    ml_model_used, ml_prediction_skipped,
                    backtest_signal, backtest_sharpe,
                    vix_regime, structural_quality, resolution_date,
                    resolution_status, resolution_attempts, created_at, updated_at, run_id
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker, brief_date) DO NOTHING
                """,
                (
                    brief.ticker,
                    today_str,
                    brief.recommendation,
                    brief.conviction,
                    ml.direction if ml else None,
                    ml.prob_up_raw if ml else None,
                    ml.cv_f1_macro if ml else None,
                    ml.cv_f1_std if ml else None,
                    ml.model_used if ml else None,
                    int(ml.prediction_skipped) if ml else 0,
                    ml.backtest_signal if ml else None,
                    ml.backtest_sharpe if ml else None,
                    brief.get_vix_regime(),
                    json.dumps(quality_metrics) if quality_metrics else None,
                    resolution_date,
                    "PENDING",
                    0,
                    now,
                    now,
                    _RUN_ID,
                ),
            )
            conn.commit()
            if cursor.rowcount == 0:
                _skipped_reruns += 1
                logger.warning(
                    "agent_eval row for %s on %s already exists, rerun skipped (%d skipped in this process)",
                    brief.ticker,
                    today_str,
                    _skipped_reruns,
                )
        finally:
            conn.close()


def resolve_pending_outcomes() -> int:
    """Fetch actual outcomes for matured PENDING rows. Returns count of resolved rows."""
    _init_db()
    today_str = date.today().isoformat()
    resolved = 0

    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT id, ticker, brief_date, recommendation, ml_direction, resolution_date,
                       resolution_attempts
                FROM agent_eval
                WHERE resolution_status = 'PENDING'
                  AND resolution_date <= ?
                  AND resolution_attempts < ?
                """,
                (today_str, MAX_RESOLUTION_ATTEMPTS),
            ).fetchall()
        finally:
            conn.close()

    for row in rows:
        row_id = row["id"]
        ticker = row["ticker"]
        attempts = row["resolution_attempts"]

        try:
            resolution_date = date.fromisoformat(row["resolution_date"])
            hist = yf.download(
                ticker,
                start=resolution_date,
                end=add_trading_days(resolution_date, 3),
                progress=False,
                auto_adjust=True,
            )
            if hist.empty:
                raise ValueError("empty yfinance response")

            if hasattr(hist.columns, "levels"):
                hist.columns = hist.columns.get_level_values(0)

            brief_hist = yf.download(
                ticker,
                start=date.fromisoformat(row["brief_date"]),
                end=add_trading_days(date.fromisoformat(row["brief_date"]), 3),
                progress=False,
                auto_adjust=True,
            )
            if hasattr(brief_hist.columns, "levels"):
                brief_hist.columns = brief_hist.columns.get_level_values(0)

            brief_close = float(brief_hist["Close"].iloc[0])
            resolution_close = float(hist["Close"].iloc[0])
            actual_return = resolution_close / brief_close - 1
            actual_direction = "UP" if actual_return > 0 else "DOWN"

            synth_correct = compute_synth_correct(row["recommendation"], actual_direction)
            ml_correct: bool | None = None
            if row["ml_direction"]:
                ml_correct = row["ml_direction"] == actual_direction

            now = time.time()
            with _DB_LOCK:
                conn = _connect()
                try:
                    conn.execute(
                        """
                        UPDATE agent_eval SET
                            actual_direction=?, actual_5d_return=?,
                            synth_correct=?, ml_correct=?,
                            resolution_status='RESOLVED', updated_at=?
                        WHERE id=?
                        """,
                        (
                            actual_direction,
                            actual_return,
                            None if synth_correct is None else int(synth_correct),
                            None if ml_correct is None else int(ml_correct),
                            now,
                            row_id,
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()
            resolved += 1

        except Exception as exc:
            logger.warning("Resolution failed for row %d (%s): %s", row_id, ticker, exc)
            new_attempts = attempts + 1
            status = "UNRESOLVABLE" if new_attempts >= MAX_RESOLUTION_ATTEMPTS else "PENDING"
            with _DB_LOCK:
                conn = _connect()
                try:
                    conn.execute(
                        "UPDATE agent_eval SET resolution_attempts=?, resolution_status=?, updated_at=? WHERE id=?",
                        (new_attempts, status, time.time(), row_id),
                    )
                    conn.commit()
                finally:
                    conn.close()

    return resolved


def compute_rolling_metrics(min_n: int = MIN_OOS_FOR_MEANINGFUL_ACCURACY) -> dict | None:
    """Return per-ticker accuracy over every resolved row. Returns None if any ticker is below min_n.

    The figures are cumulative: despite the name there is no date window.
    """
    _init_db()
    with _DB_LOCK:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT ticker, synth_correct, ml_correct, resolution_status "
                "FROM agent_eval WHERE resolution_status='RESOLVED'"
            ).fetchall()
            unresolvable_rows = conn.execute(
                "SELECT ticker, COUNT(*) FROM agent_eval WHERE resolution_status='UNRESOLVABLE' GROUP BY ticker"
            ).fetchall()
        finally:
            conn.close()
    unresolvable_by_ticker = {r[0]: int(r[1]) for r in unresolvable_rows}

    if not rows:
        return None

    by_ticker: dict[str, list] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(row)

    if any(len(v) < min_n for v in by_ticker.values()):
        return None

    result = {}
    for ticker, ticker_rows in by_ticker.items():
        synth = [r["synth_correct"] for r in ticker_rows if r["synth_correct"] is not None]
        ml = [r["ml_correct"] for r in ticker_rows if r["ml_correct"] is not None]
        unresolvable = unresolvable_by_ticker.get(ticker, 0)
        result[ticker] = {
            "n": len(ticker_rows),
            "synth_accuracy": sum(synth) / len(synth) if synth else None,
            "ml_accuracy": sum(ml) / len(ml) if ml else None,
            "unresolvable_fraction": unresolvable / (len(ticker_rows) + unresolvable),
            "ready_for_decisions": len(ticker_rows) >= MIN_OOS_FOR_DECISIONS,
        }
    return result


def should_trigger_review() -> dict[str, bool]:
    """Return trigger flags. All False until BASELINE_WINDOW_WEEKS of data per ticker.

    Note: agent_underperforming is intentionally excluded - synth_correct evaluates
    a weeks-to-months thesis against a 5-day window (horizon mismatch).
    Use it only as an observability signal, not a replacement trigger.

    ml_drift_detected is a fixed accuracy floor (0.5 minus DRIFT_STD_MULTIPLIER times 0.1), not a comparison
    with an estimated baseline; BASELINE_WINDOW_WEEKS only sets the row count before any flag can fire.
    """
    metrics = compute_rolling_metrics(min_n=MIN_OOS_FOR_MEANINGFUL_ACCURACY)

    baseline_false: dict[str, bool] = {
        "insufficient_data": False,
        "ml_drift_detected": False,
        "both_degraded": False,
    }

    if metrics is None:
        return baseline_false

    # Suppress all flags until BASELINE_WINDOW_WEEKS of data per ticker
    min_n_for_baseline = int(BASELINE_WINDOW_WEEKS * 5)  # ~5 trading days per week
    if any(v["n"] < min_n_for_baseline for v in metrics.values()):
        return baseline_false

    insufficient_data = any(not v["ready_for_decisions"] for v in metrics.values())
    ml_drift = False
    for ticker_metrics in metrics.values():
        acc = ticker_metrics.get("ml_accuracy")
        if acc is not None and acc < 0.5 - DRIFT_STD_MULTIPLIER * 0.1:
            ml_drift = True

    return {
        "insufficient_data": insufficient_data,
        "ml_drift_detected": ml_drift,
        "both_degraded": ml_drift and insufficient_data,
    }
