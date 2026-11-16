"""Run mypy and fail when the error count rises above the recorded ceiling.

The package still has known typing errors. This check stops the count from growing: lower MAX_ERRORS
whenever the count falls, and never raise it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_ERRORS = 62
_FOUND = re.compile(r"^Found (\d+) errors? in \d+ files?", re.M)
_CLEAN = re.compile(r"^Success: no issues found", re.M)


def error_count(output: str) -> int | None:
    """Number of errors in mypy's summary line, or None when the output carries no summary."""
    found = _FOUND.search(output)
    if found:
        return int(found.group(1))
    if _CLEAN.search(output):
        return 0
    return None


def verdict(count: int | None, ceiling: int = MAX_ERRORS) -> tuple[int, str]:
    if count is None:
        return 2, "mypy_ratchet: no summary line in the mypy output, so the count is unknown"
    if count > ceiling:
        return 1, f"mypy_ratchet: {count} errors, above the ceiling of {ceiling}"
    if count < ceiling:
        return 0, f"mypy_ratchet: {count} errors, below the ceiling of {ceiling}; lower MAX_ERRORS to {count}"
    return 0, f"mypy_ratchet: {count} errors, at the ceiling of {ceiling}"


def main() -> int:
    run = subprocess.run(
        [sys.executable, "-m", "mypy"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        cwd=ROOT,
    )
    output = run.stdout + run.stderr
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(output, end="" if output.endswith("\n") else "\n")
    code, message = verdict(error_count(output))
    print(message)
    return code


if __name__ == "__main__":
    sys.exit(main())
