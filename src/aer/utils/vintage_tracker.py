"""Thread-safe vintage timestamp tracking for per-field data freshness.

TODO: Add audit log for vintage changes:
- Log entry when fetch timestamp is overwritten (debug: which source, old → new time)
- Write audit log to: a path under the cache directory (per-session)
- Include session_id, ticker, source, old_timestamp, new_timestamp, reason
- Use for debugging: identify sources being refetched vs. cached
- Also useful for identifying cache misses or refresh patterns
- Max 1000 entries per session; rotate on overflow
"""

import logging
import threading
import time
from typing import Final

logger = logging.getLogger(__name__)

# Mapping from data source domain to brief.vintage key name
SOURCE_TO_VINTAGE_KEY: Final[dict[str, str]] = {
    "prices": "prices",
    "consensus": "consensus",
    "macro": "macro",
    "earnings": "earnings",
    "ownership": "ownership",
    "options": "options",
    "sentiment": "sentiment",
    "factors": "factors",
}

# Thread-local storage for vintage timestamps per build session
_session_vintage: threading.local = threading.local()


def start_session() -> None:
    """Initialize a new vintage tracking session. Call once per invoke_pipeline() invocation."""
    _session_vintage.timestamps = {}


def record_vintage(source: str, timestamp: float | None = None) -> None:
    """Record the fetch timestamp for a data source.

    Keeps the MOST RECENT (maximum) timestamp per source in the current session.
    If called multiple times for the same source (e.g., retry), the latest timestamp
    is retained. This ensures brief.vintage reflects the actual fetch time.

    Args:
        source: One of the keys in SOURCE_TO_VINTAGE_KEY (e.g., 'prices', 'consensus')
        timestamp: Unix timestamp (float). If None, uses current time (time.time()).
    """
    if not hasattr(_session_vintage, "timestamps"):
        _session_vintage.timestamps = {}

    if timestamp is None:
        timestamp = time.time()

    key = SOURCE_TO_VINTAGE_KEY.get(source)
    if key is None:
        logger.warning("Unknown vintage source: %s", source)
        return

    # Keep the most recent (maximum) timestamp
    if key not in _session_vintage.timestamps:
        _session_vintage.timestamps[key] = timestamp
    else:
        # Update if this timestamp is more recent than the existing one
        if timestamp > _session_vintage.timestamps[key]:
            _session_vintage.timestamps[key] = timestamp


def get_session_vintage() -> dict[str, float]:
    """Retrieve the collected vintage timestamps for the current session.

    Returns:
        Dict mapping vintage key to Unix timestamp (float), or empty dict if no session active.
    """
    if not hasattr(_session_vintage, "timestamps"):
        return {}
    return _session_vintage.timestamps.copy()


def clear_session() -> None:
    """Clear the current session's vintage tracking."""
    _session_vintage.timestamps = {}


def record_multiple(source_timestamps: dict[str, float]) -> None:
    """Record multiple vintage timestamps at once.

    Args:
        source_timestamps: Dict mapping source name to Unix timestamp.
    """
    for source, ts in source_timestamps.items():
        record_vintage(source, ts)
