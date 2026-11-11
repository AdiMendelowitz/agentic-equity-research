"""SQLite-backed cache with per-entry TTL and recommendation audit log.

WAL mode for concurrent reads; double-checked locking on the get_cache() singleton.
"""

import json
import logging
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aer.paths import PROJECT_ROOT as _PROJECT_ROOT
from aer.utils.db import connect_cache_db

logger = logging.getLogger(__name__)

TTL_PRICES: int = 24 * 3600
TTL_QUOTE: int = 1 * 3600
TTL_ANALYSIS: int = 12 * 3600
TTL_BRIEF: int = TTL_ANALYSIS
TTL_NEWS: int = 3600  # 1 hour - news goes stale fast; market-aware extension not applicable

# Prediction Markets + Social Intelligence
TTL_PREDICTION_MARKETS: int = 6 * 3600  # 6h - Polymarket/PredictIt odds
TTL_SOCIAL_INTELLIGENCE: int = 6 * 3600  # 6h - aggregated signals (Layer 1)
TTL_SOCIAL_SYNTHESIS: int = 24 * 3600  # 24h - optional Layer 2 LLM synthesis

# LSEG-specific TTLs
TTL_CONSENSUS: int = 14_400  # 4h - IBES revised pre-market daily, sometimes intraday on earnings day
TTL_LSEG_FUNDAMENTALS: int = 86_400  # 24h - quarterly source; refresh daily to catch restatements
TTL_LSEG_MACRO: int = 21_600  # 6h - rate curves and swap rates update intraday


def market_aware_ttl(base_ttl: int, _now: "datetime | None" = None) -> int:
    """Extend base_ttl so the cache entry does not expire on a weekend or NYSE holiday.

    If the nominal expiry date falls on a non-trading day, the TTL is stretched to
    9:30 AM ET (14:30 UTC) on the next trading day so stale data is never surfaced
    during market closure.

    _now is injected only in tests; production callers omit it.
    """
    from datetime import datetime, timedelta

    import pandas as pd
    from pandas.tseries.holiday import USFederalHolidayCalendar

    now = _now if _now is not None else datetime.now(tz=UTC)
    nominal_expiry = now + timedelta(seconds=base_ttl)
    expiry_date = nominal_expiry.date()

    # USFederalHolidayCalendar misses NYSE-specific closures (Good Friday) and
    # includes federal holidays NYSE does not observe (Columbus Day, Veterans Day).
    # Approximation is acceptable; upgrade to exchange_calendars.NYSE when available.
    holidays = USFederalHolidayCalendar().holidays(
        start=expiry_date.isoformat(),
        end=(expiry_date + timedelta(days=10)).isoformat(),
    )
    logger.debug("market_aware_ttl: using USFederalHolidayCalendar (NYSE approximation)")

    expiry_ts = pd.Timestamp(expiry_date)
    advanced = False
    while expiry_ts.day_of_week >= 5 or expiry_ts in holidays:
        expiry_ts += pd.Timedelta(days=1)
        advanced = True

    if not advanced:
        return base_ttl

    extended = datetime(expiry_ts.year, expiry_ts.month, expiry_ts.day, 14, 30, tzinfo=UTC)
    return max(base_ttl, int((extended - now).total_seconds()))


_MISSING: object = object()
"""Sentinel returned by Cache.get() when a key is absent or expired.

Callers must check `value is _MISSING` — NOT `value is not None` — so that
legitimate None/null values stored in cache are correctly distinguished from misses.
This distinction is critical for agents that may legitimately cache None (e.g., a None recommendation)
versus returning early when a cache miss occurs (no data available).
"""


class Cache:
    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            from aer.config import get_settings

            db_path = get_settings().cache_db_path

        path = Path(db_path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)

        self._path = str(path)
        self._local = threading.local()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = connect_cache_db(self._path)
            self._local.conn = conn
        return self._local.conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key         TEXT    PRIMARY KEY,
                value       TEXT    NOT NULL,
                ttl_seconds INTEGER NOT NULL,
                created_at  REAL    NOT NULL,
                domain      TEXT    NOT NULL DEFAULT 'legacy'
            )
            """)
        conn.execute("""
 CREATE TABLE IF NOT EXISTS recommendations (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 ticker TEXT NOT NULL,
 recommendation TEXT NOT NULL,
 conviction REAL NOT NULL,
 price REAL,
 logged_at REAL NOT NULL,
 sector TEXT -- sector for cross-ticker scan, added by ALTER TABLE on existing DBs
 )
 """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_ticker ON recommendations(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_logged_at ON recommendations(logged_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_domain ON cache(domain)")
        try:
            conn.execute("ALTER TABLE cache ADD COLUMN domain TEXT NOT NULL DEFAULT 'legacy'")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)
            """)
        conn.execute("INSERT OR IGNORE INTO schema_version VALUES (1)")
        # Opinion tracking tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS opinions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                opinion_date TEXT NOT NULL,
                signal TEXT NOT NULL CHECK(signal IN ('BUY', 'HOLD', 'SELL', 'AVOID')),
                confidence REAL NOT NULL CHECK(confidence BETWEEN 0.0 AND 1.0),
                conviction REAL CHECK(conviction IS NULL OR conviction BETWEEN 0.0 AND 1.0),
                drivers TEXT,
                logged_at REAL NOT NULL,
                forward_return_5d REAL,
                UNIQUE(ticker, agent_name, opinion_date),
                CHECK(logged_at > 0)
            )
            """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_opinions_ticker ON opinions(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_opinions_agent ON opinions(agent_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_opinions_date ON opinions(opinion_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_opinions_logged_at ON opinions(logged_at)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS opinion_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                ticker TEXT NOT NULL,
                total_opinions INTEGER NOT NULL,
                binary_hit_rate REAL,
                return_weighted_score REAL,
                sharpe_ratio REAL,
                conviction_weighted_hit_rate REAL,
                n_profitable_holds INTEGER DEFAULT 0,
                last_opinion_date TEXT,
                insufficient_data INTEGER DEFAULT 0,
                computed_at REAL NOT NULL,
                UNIQUE(agent_name, ticker),
                CHECK(computed_at > 0)
            )
            """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_opinion_metrics_agent ON opinion_metrics(agent_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_opinion_metrics_ticker ON opinion_metrics(ticker)")
        conn.commit()

    def get(self, key: str) -> Any:
        """Return the cached value, or _MISSING if the key is absent or expired.

        Callers must check `result is _MISSING` - NOT `result is not None` - so that
        legitimate stored None values are correctly distinguished from cache misses.

        Uses a single atomic SQL statement to check expiry and avoid race conditions.
        """
        conn = self._connect()
        # Single atomic check: SELECT only if NOT expired. Avoids TOCTOU gap.
        # created_at is stored in milliseconds, so convert now to ms and convert
        # age back to seconds for comparison with ttl_seconds
        now_ms = time.time() * 1000
        row = conn.execute(
            "SELECT value FROM cache WHERE key = ? AND ((? - created_at) / 1000.0) <= ttl_seconds",
            (key, now_ms),
        ).fetchone()

        if row is None:
            return _MISSING

        (value_json,) = row
        return json.loads(value_json)

    def set(self, key: str, value: Any, ttl: int = TTL_PRICES) -> None:  # noqa: ARG002
        raise RuntimeError(
            "Call _DomainCache.set() — direct Cache.set() bypasses domain tagging. "
            "Use data_cache(), agent_cache(), or brief_cache() from aer.utils.cache."
        )

    def _set_with_domain(self, key: str, value: Any, ttl: int, domain: str) -> None:
        value_json = json.dumps(value, default=str)
        conn = self._connect()
        # Use millisecond-precision timestamps to distinguish rapid successive writes
        created_at_ms = time.time() * 1000
        conn.execute(
            """
            INSERT INTO cache (key, value, ttl_seconds, created_at, domain)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                ttl_seconds = excluded.ttl_seconds,
                created_at = excluded.created_at,
                domain = excluded.domain
            """,
            (key, value_json, ttl, created_at_ms, domain),
        )
        conn.commit()

    def delete(self, key: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()

    def cleanup(self) -> int:
        # created_at is in milliseconds, convert now to ms for comparison
        now_ms = time.time() * 1000
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM cache WHERE ((? - created_at) / 1000.0) > ttl_seconds",
            (now_ms,),
        )
        conn.commit()
        return cursor.rowcount

    def invalidate_prefix(self, prefix: str) -> int:
        """Delete cache entries matching a key prefix.

        WARNING: This base method invalidates across all domains. For domain-specific
        invalidation, use _DomainCache via aer.utils.cache module instead.
        """
        conn = self._connect()
        cursor = conn.execute("DELETE FROM cache WHERE key LIKE ?", (f"{prefix}%",))
        conn.commit()
        logger.warning(
            "Cache.invalidate_prefix(%r) called on base cache — this affects ALL domains. "
            "Use aer.utils.cache domain caches (brief_cache(), agent_cache(), etc.) for domain-specific invalidation.",
            prefix,
        )
        return cursor.rowcount

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not _MISSING

    def ttl_for(self, key: str) -> int | None:
        """Return the stored ttl_seconds for key, or None if key does not exist."""
        conn = self._connect()
        row = conn.execute("SELECT ttl_seconds FROM cache WHERE key = ?", (key,)).fetchone()
        return int(row[0]) if row is not None else None

    def created_at_for(self, key: str) -> float | None:
        """Return the created_at Unix timestamp for key, or None if key does not exist or is expired."""
        conn = self._connect()
        row = conn.execute("SELECT created_at, ttl_seconds FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        created_at_ms, ttl = row
        # created_at is in milliseconds, convert to seconds for comparison
        age_seconds = (time.time() * 1000 - created_at_ms) / 1000.0
        if age_seconds > ttl:
            return None
        return float(created_at_ms) / 1000.0  # Return in seconds

    def keys_matching(self, pattern: str) -> list[str]:
        """Return all cache keys matching an SQL LIKE pattern (e.g. 'brief:AAPL:%')."""
        conn = self._connect()
        rows = conn.execute("SELECT key FROM cache WHERE key LIKE ?", (pattern,)).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Recommendation audit log
    # ------------------------------------------------------------------

    def log_recommendation(
        self,
        ticker: str,
        recommendation: str,
        conviction: float,
        price: float | None,
    ) -> None:
        """Append a recommendation to the audit log. Never overwrites; always appends."""
        conn = self._connect()
        conn.execute(
            "INSERT INTO recommendations (ticker, recommendation, conviction, price, logged_at) VALUES (?,?,?,?,?)",
            (ticker.upper(), recommendation, conviction, price, time.time()),
        )
        conn.commit()

    def get_recommendation_log(self, days: int = 90) -> list[dict[str, Any]]:
        """Return all recommendations logged within the last `days` days, newest first."""
        cutoff = time.time() - days * 86400
        conn = self._connect()
        rows = conn.execute(
            "SELECT ticker, recommendation, conviction, price, logged_at FROM recommendations "
            "WHERE logged_at >= ? ORDER BY logged_at DESC",
            (cutoff,),
        ).fetchall()
        return [
            {
                "ticker": r[0],
                "recommendation": r[1],
                "conviction": r[2],
                "price": r[3],
                "logged_at": datetime.fromtimestamp(r[4], tz=UTC).isoformat(),
                "logged_at_ts": r[4],
            }
            for r in rows
        ]

    def upsert_metrics(self, metrics: Any) -> None:
        """Insert or update opinion_metrics row. Called by nightly job."""
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO opinion_metrics
                (agent_name, ticker, total_opinions, binary_hit_rate, return_weighted_score,
                 sharpe_ratio, conviction_weighted_hit_rate, last_opinion_date,
                 n_profitable_holds, insufficient_data, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_name, ticker) DO UPDATE SET
                    total_opinions=excluded.total_opinions,
                    binary_hit_rate=excluded.binary_hit_rate,
                    return_weighted_score=excluded.return_weighted_score,
                    sharpe_ratio=excluded.sharpe_ratio,
                    conviction_weighted_hit_rate=excluded.conviction_weighted_hit_rate,
                    last_opinion_date=excluded.last_opinion_date,
                    n_profitable_holds=excluded.n_profitable_holds,
                    insufficient_data=excluded.insufficient_data,
                    computed_at=excluded.computed_at
                """,
                (
                    metrics.agent_name,
                    metrics.ticker,
                    metrics.total_opinions,
                    metrics.binary_hit_rate,
                    metrics.return_weighted_score,
                    metrics.sharpe_ratio,
                    metrics.conviction_weighted_hit_rate,
                    metrics.last_opinion_date.isoformat() if metrics.last_opinion_date else None,
                    metrics.n_profitable_holds,
                    1 if metrics.insufficient_data else 0,
                    time.time(),
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error("Failed to upsert metrics: %s", e)
            raise

    def get_metrics(self, agent_name: str | None = None, ticker: str | None = None) -> list[Any]:
        """Retrieve cached metrics. Used by discovery page."""
        from datetime import date

        from aer.models import HitRateMetrics

        conn = self._connect()

        query = (
            "SELECT agent_name, ticker, total_opinions, binary_hit_rate, return_weighted_score, "
            "sharpe_ratio, conviction_weighted_hit_rate, last_opinion_date, n_profitable_holds, "
            "insufficient_data FROM opinion_metrics WHERE 1=1"
        )
        params: list[Any] = []

        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker.upper())

        query += " ORDER BY computed_at DESC"
        rows = conn.execute(query, params).fetchall()

        metrics_list = []
        for row in rows:
            m = HitRateMetrics(
                agent_name=row[0],
                ticker=row[1],
                total_opinions=row[2],
                binary_hit_rate=row[3],
                return_weighted_score=row[4],
                sharpe_ratio=row[5],
                conviction_weighted_hit_rate=row[6],
                last_opinion_date=date.fromisoformat(row[7]) if row[7] else None,
                n_profitable_holds=row[8],
                insufficient_data=bool(row[9]),
            )
            metrics_list.append(m)

        return metrics_list


_default_cache: Cache | None = None
_cache_lock = threading.Lock()


def get_cache() -> Cache:
    global _default_cache
    if _default_cache is None:
        with _cache_lock:
            if _default_cache is None:
                _default_cache = Cache()
    return _default_cache


def reset_cache() -> None:
    """Reset the global cache instance. For testing only."""
    global _default_cache
    with _cache_lock:
        _default_cache = None
