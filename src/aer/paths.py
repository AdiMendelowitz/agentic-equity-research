"""Single source of filesystem locations for the package. Every module resolves paths from here, never from its own depth."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_ROOT.parent.parent
RESOURCES_DIR: Path = PACKAGE_ROOT / "resources"
ENV_FILE: Path = PROJECT_ROOT / ".env"
WATCHLIST_FILE: Path = PROJECT_ROOT / "watchlist.json"
CACHE_DIR: Path = Path(os.environ.get("AER_CACHE_DIR", PROJECT_ROOT / ".cache"))
ARTIFACT_DIR_ENV: str = "AER_ARTIFACT_DIR"
