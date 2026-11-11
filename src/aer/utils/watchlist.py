"""Watchlist I/O with BOM handling and loud failure on missing/unparseable files."""

import json
import logging
import os
from pathlib import Path

from aer.paths import WATCHLIST_FILE as _WATCHLIST_PATH
from aer.utils.constants import WATCHLIST_MAX_TICKERS

logger = logging.getLogger(__name__)


_DEFAULT_WATCHLIST: list[str] = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AMD",
    "JPM",
    "BRK-B",
]


class WatchlistLoadError(RuntimeError):
    """Raised when watchlist cannot be loaded (missing, unparseable, or I/O error)."""

    pass


def load_watchlist(path: Path | None = None) -> list[str]:
    """Load watchlist from JSON with BOM-safe UTF-8 handling.

    Args:
    path: Explicit file to read. Defaults to the module watchlist.json.
    Tests MUST inject a path rather than patch the module global.

    Raises WatchlistLoadError if file is missing, unparseable, or unreadable.
    """
    target = path if path is not None else _WATCHLIST_PATH
    try:
        if not target.exists():
            raise FileNotFoundError(f"Watchlist file not found: {target}")
        data = json.loads(target.read_text(encoding="utf-8-sig"))
        result = [t for t in data if isinstance(t, str)][:WATCHLIST_MAX_TICKERS]
        return result
    except (OSError, FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to load watchlist from %s: %s", target, exc)
        raise WatchlistLoadError(f"Unable to load watchlist: {exc}") from exc


def save_watchlist(tickers: list[str], path: Path | None = None) -> None:
    """Save watchlist to JSON with UTF-8 (no BOM) encoding.

    Args:
    tickers: List of ticker symbols (truncated to WATCHLIST_MAX_TICKERS).
    path: Explicit file to write. Defaults to the module watchlist.json.
    Tests MUST inject a path rather than patch the module global.
    """
    target = path if path is not None else _WATCHLIST_PATH
    try:
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(tickers[:WATCHLIST_MAX_TICKERS], indent=2), encoding="utf-8")
        os.replace(tmp, target)
    except OSError as e:
        logger.error("Failed to save watchlist to %s: %s", target, e)
        raise
