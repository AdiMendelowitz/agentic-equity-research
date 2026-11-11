"""Project cache facade — single import point for all callers.

Backed by src/data/cache.SQLite store. New modules must import from here,
not from aer.data.cache, so the surface stays auditable.
"""

import logging
import sys as _sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from aer.data.cache import (
    _MISSING,
    TTL_ANALYSIS,
    TTL_BRIEF,
    TTL_CONSENSUS,
    TTL_LSEG_FUNDAMENTALS,
    TTL_LSEG_MACRO,
    TTL_NEWS,
    TTL_PRICES,
    TTL_QUOTE,
    Cache,
    get_cache,
    market_aware_ttl,
)

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)


class _DomainCache:
    """Domain-scoped cache wrapper. Routes set() through _set_with_domain() to tag rows."""

    def __init__(
        self,
        cache: Cache,
        default_ttl: "int | Callable[[], int]",
        domain: str,
    ) -> None:
        self._cache = cache
        self._default_ttl = default_ttl
        self._domain = domain

    def get(self, key: str) -> Any:
        """Return cached value or None if absent/expired.

        Returns None for both cache misses and stored None values — backward compatible
        with all existing callers that check `if cached is not None`.
        Use get_or_compute() when you need to distinguish a stored None from a miss.
        """
        result = self._cache.get(key)
        return None if result is _MISSING else result

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        resolved_ttl = (
            ttl if ttl is not None else (self._default_ttl() if callable(self._default_ttl) else self._default_ttl)
        )
        self._cache._set_with_domain(key, value, resolved_ttl, self._domain)

    def delete(self, key: str) -> None:
        self._cache.delete(key)

    def cleanup(self) -> int:
        """Delete expired entries in this domain only. Returns rows deleted.
        For global cleanup across all domains, call Cache.cleanup() on the underlying cache."""
        import time as _time

        # created_at is in milliseconds, convert now to ms for comparison
        now_ms = _time.time() * 1000
        conn = self._cache._connect()
        cursor = conn.execute(
            "DELETE FROM cache WHERE ((? - created_at) / 1000.0) > ttl_seconds AND domain = ?",
            (now_ms, self._domain),
        )
        conn.commit()
        return cursor.rowcount

    def flush(self) -> int:
        conn = self._cache._connect()
        cursor = conn.execute("DELETE FROM cache WHERE domain = ?", (self._domain,))
        conn.commit()
        return cursor.rowcount

    def invalidate_prefix(self, prefix: str) -> int:
        conn = self._cache._connect()
        cursor = conn.execute(
            "DELETE FROM cache WHERE key LIKE ? AND domain = ?",
            (f"{prefix}%", self._domain),
        )
        conn.commit()
        return cursor.rowcount

    def __contains__(self, key: str) -> bool:
        import time as _time

        now_ms = _time.time() * 1000
        conn = self._cache._connect()
        row = conn.execute(
            "SELECT created_at, ttl_seconds FROM cache WHERE key = ? AND domain = ?",
            (key, self._domain),
        ).fetchone()
        if row is None:
            return False
        created_at, ttl = row
        return ((now_ms - created_at) / 1000.0) <= ttl

    def created_at_for(self, key: str) -> float | None:
        return self._cache.created_at_for(key)


class RecommendationLog:
    """Audit log for recommendation history. Wraps Cache.log_recommendation."""

    def __init__(self, cache: Cache) -> None:
        self._cache = cache

    def log(
        self,
        ticker: str,
        recommendation: str,
        conviction: float,
        price: float | None,
    ) -> None:
        self._cache.log_recommendation(ticker, recommendation, conviction, price)

    def get_log(self, days: int = 90) -> list[dict[str, Any]]:
        return self._cache.get_recommendation_log(days)


__all__ = [
    "Cache",
    "TTL_ANALYSIS",
    "TTL_BRIEF",
    "TTL_CONSENSUS",
    "TTL_LSEG_FUNDAMENTALS",
    "TTL_LSEG_MACRO",
    "TTL_NEWS",
    "TTL_PRICES",
    "TTL_QUOTE",
    "_DomainCache",
    "RecommendationLog",
    "CacheComputeError",
    "data_cache",
    "agent_cache",
    "brief_cache",
    "news_cache",
    "lseg_cache",
    "get_recommendation_log_store",
    "get_or_compute",
    "market_aware_ttl",
    "get_professor_scores",
]

_data_cache: _DomainCache | None = None
_agent_cache: _DomainCache | None = None
_brief_cache: _DomainCache | None = None
_rec_log: RecommendationLog | None = None
_data_lock = threading.Lock()
_agent_lock = threading.Lock()
_brief_lock = threading.Lock()
_rec_lock = threading.Lock()
_news_cache: _DomainCache | None = None
_news_lock = threading.Lock()
_lseg_cache: _DomainCache | None = None
_lseg_lock = threading.Lock()


def data_cache() -> _DomainCache:
    global _data_cache
    if _data_cache is None:
        with _data_lock:
            if _data_cache is None:
                _data_cache = _DomainCache(get_cache(), TTL_PRICES, "data")
    return _data_cache


def agent_cache() -> _DomainCache:
    global _agent_cache
    if _agent_cache is None:
        with _agent_lock:
            if _agent_cache is None:
                _agent_cache = _DomainCache(get_cache(), TTL_ANALYSIS, "agent")
    return _agent_cache


def brief_cache() -> _DomainCache:
    global _brief_cache
    if _brief_cache is None:
        with _brief_lock:
            if _brief_cache is None:
                _brief_cache = _DomainCache(get_cache(), lambda: market_aware_ttl(TTL_BRIEF), "brief")
    return _brief_cache


def news_cache() -> _DomainCache:
    global _news_cache
    if _news_cache is None:
        with _news_lock:
            if _news_cache is None:
                _news_cache = _DomainCache(get_cache(), TTL_NEWS, "news")
    return _news_cache


def lseg_cache() -> _DomainCache:
    """Domain cache for LSEG MCP-sourced data.

    TTL varies by sub-type — callers should pass explicit ttl= on .set():
      - consensus signals: TTL_CONSENSUS (4h)
      - fundamentals signals: TTL_LSEG_FUNDAMENTALS (24h)
      - macro signals: TTL_LSEG_MACRO (6h)
    Default TTL (TTL_CONSENSUS) applies when ttl= is omitted.
    """
    global _lseg_cache
    if _lseg_cache is None:
        with _lseg_lock:
            if _lseg_cache is None:
                _lseg_cache = _DomainCache(get_cache(), TTL_CONSENSUS, "lseg")
    return _lseg_cache


def get_recommendation_log_store() -> RecommendationLog:
    global _rec_log
    if _rec_log is None:
        with _rec_lock:
            if _rec_log is None:
                _rec_log = RecommendationLog(get_cache())
    return _rec_log


class CacheComputeError(Exception):
    """Raised by get_or_compute() when compute_fn raises and suppress_errors is False."""


def get_or_compute(
    key: str,
    compute_fn: Callable[[], Any],
    ttl: int,
    cache: "_DomainCache | None" = None,
    suppress_errors: bool = False,
) -> Any:
    """Return cached value or call compute_fn(), cache it, and return.

    Raises CacheComputeError if compute_fn raises and suppress_errors is False (default).
    Pass suppress_errors=True to restore legacy behavior of returning None on failure.
    """
    c: _DomainCache = cache if cache is not None else data_cache()
    # Bypass _DomainCache.get() (which maps _MISSING→None for backward compat) so
    # stored None values are correctly distinguished from cache misses.
    cached = c._cache.get(key)
    if cached is not _MISSING:
        return cached
    try:
        value = compute_fn()
    except Exception as exc:
        _logger.warning("compute_fn failed for key %r: %s", key, exc)
        if suppress_errors:
            return None
        raise CacheComputeError(f"compute_fn failed for {key!r}") from exc
    c.set(key, value, ttl=ttl)
    return value


def get_professor_scores(tickers: list[str]) -> dict[str, dict | None]:
    """Batch-fetch professor_score and professor_flags for each ticker from the recommendations table.

    Returns a dict mapping ticker (uppercased) to:
      {"score": float, "flags": list[str]}  -- when professor data exists
      None                                   -- when no non-null professor_score row exists

    Uses a single sqlite3 connection with one IN-clause query for all tickers.
    Reads from get_settings().cache_db_path (same DB as _write_to_recommendation_log).
    """
    import json as _json
    import sqlite3 as _sqlite3

    from aer.config import get_settings
    from aer.utils.db import connect_cache_db

    if not tickers:
        return {}

    upper = [t.upper() for t in tickers]
    result: dict[str, dict | None] = {t: None for t in upper}

    db_path = get_settings().cache_db_path
    try:
        conn = connect_cache_db(str(db_path))
        try:
            placeholders = ",".join("?" * len(upper))
            rows = conn.execute(
                f"SELECT ticker, professor_score, professor_flags "
                f"FROM recommendations "
                f"WHERE ticker IN ({placeholders}) AND professor_score IS NOT NULL "
                f"AND id IN ("
                f"  SELECT MAX(id) FROM recommendations "
                f"  WHERE ticker IN ({placeholders}) AND professor_score IS NOT NULL "
                f"  GROUP BY ticker"
                f")",
                upper + upper,
            ).fetchall()
        finally:
            conn.close()
    except _sqlite3.OperationalError:
        # Expected on a fresh install before first professor critique run (table not yet created).
        return result
    except Exception as exc:
        _logger.error("get_professor_scores query failed unexpectedly: %s", exc)
        return result

    for ticker, score, flags_json in rows:
        flags = _json.loads(flags_json) if flags_json else []
        result[ticker.upper()] = {"score": float(score), "flags": flags}

    return result


def _flush_cli() -> None:
    """Entry point for: python -m aer.utils.cache flush [data|agent|brief|news|lseg|all]"""
    args = _sys.argv[1:]
    if not args or args[0] != "flush":
        print("Usage: python -m aer.utils.cache flush [data|agent|brief|news|lseg|all]")
        _sys.exit(1)

    domain_arg = args[1] if len(args) > 1 else ""
    valid = {"data", "agent", "brief", "news", "lseg", "all"}

    if not domain_arg or domain_arg not in valid:
        print("Usage: python -m aer.utils.cache flush [data|agent|brief|news|lseg|all]")
        _sys.exit(1)

    if domain_arg == "all":
        for name, accessor in [
            ("data", data_cache),
            ("agent", agent_cache),
            ("brief", brief_cache),
            ("news", news_cache),
            ("lseg", lseg_cache),
        ]:
            n = accessor().flush()
            print(f"Flushed {n} entries from {name} cache.")
    else:
        accessor_map = {
            "data": data_cache,
            "agent": agent_cache,
            "brief": brief_cache,
            "news": news_cache,
            "lseg": lseg_cache,
        }
        n = accessor_map[domain_arg]().flush()
        print(f"Flushed {n} entries from {domain_arg} cache.")

    _sys.exit(0)


if __name__ == "__main__":
    _flush_cli()
