"""SQLite connection manager with WAL mode and busy timeout.

Sole sanctioned home of ``sqlite3.connect`` in ``src/`` every domain module
opens its DB through ``connect_db`` (or the cache-specific ``connect_cache_db``), so
WAL/busy-timeout/synchronous pragmas are applied uniformly instead of copy-pasted.
"""

import sqlite3
from collections.abc import Callable
from typing import Final

from aer.config import get_settings

SQLITE_BUSY_TIMEOUT_MS: Final[int] = 30000
"""Busy timeout in ms. Restores pre-migration Cache timeout=30.0."""


def connect_db(
    db_path: str,
    *,
    check_same_thread: bool = True,
    timeout: float | None = None,
    row_factory: Callable[[sqlite3.Cursor, tuple], object] | None = None,
    busy_timeout_ms: int = SQLITE_BUSY_TIMEOUT_MS,
) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection with a busy timeout and NORMAL synchronous.

    Domain-neutral: each caller passes its own DB path and the connect-level options
    (``check_same_thread``, ``timeout``, ``row_factory``, ``busy_timeout_ms``) it
    needs, so per-domain semantics are preserved while the pragma set is centralized.
    """
    kwargs: dict[str, object] = {"check_same_thread": check_same_thread}
    if timeout is not None:
        kwargs["timeout"] = timeout
    conn = sqlite3.connect(db_path, **kwargs)
    conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    if row_factory is not None:
        conn.row_factory = row_factory
    return conn


def connect_cache_db(db_path: str | None = None) -> sqlite3.Connection:
    """Sole sanctioned connector for the cache DB.

    db_path=None -> settings cache_db_path (production, alerts engine).
    Explicit db_path -> honored: Cache(db_path=...), test fixtures.
    check_same_thread=False restores the deliberate pre-migration setting
    for Streamlit/prefetch cross-thread access.
    """
    conn = connect_db(
        db_path if db_path is not None else str(get_settings().cache_db_path),
        check_same_thread=False,
    )
    conn.execute("PRAGMA locking_mode = NORMAL")
    return conn
