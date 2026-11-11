"""Single-instance lockfile for the pipeline and simulator entry points.

Public API (B2b plan §"New files"):
    LockHeldError
    pid_running(pid) -> bool
    acquire_lock(lock_path, stale_hours) -> None
    release_lock(lock_path) -> None

A lock is an ``O_EXCL``-created JSON file holding the owner PID, an ISO-8601 UTC timestamp, and
the hostname. Acquisition is refused if the lock is held by a live process on this host, unless
the lock is stale — stale means the owner PID is dead OR the lock is older than ``stale_hours``.
A stale lock is reclaimed with a WARNING; a live lock is refused with a message (also logged at
ERROR) naming the lockfile path, its contents, and manual-removal instructions.

Liveness uses ``ctypes.windll.kernel32.OpenProcess`` on Windows and ``os.kill(pid, 0)`` on POSIX.
No ``psutil`` (B2b corrections v4 §"Dependency confirmation" binds last over clarifications v2
§4, which had allowed psutil). Liveness is only meaningful on the host that wrote the lock; a
lock owned by another hostname is treated as live (refused) unless it has aged past ``stale_hours``.

Caveat: PID reuse can make a dead owner look live within ``stale_hours``; the age bound caps that
window. An OS advisory lock (``portalocker`` is a dependency) would be stronger, but the gate
spec mandates the PID-file mechanism.
"""

from __future__ import annotations

import ctypes
import errno
import json
import logging
import os
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_WIN_SYNCHRONIZE = 0x00100000
_WIN_WAIT_TIMEOUT = 0x00000102  # process still running
_WIN_ERROR_ACCESS_DENIED = 5  # process exists, no rights to query -> alive


class LockHeldError(RuntimeError):
    """Raised when the lock is held by a live process on this host and is not stale."""


def _pid_running_windows(pid: int) -> bool:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]  # Windows-only
    handle = kernel32.OpenProcess(_WIN_SYNCHRONIZE, False, pid)
    if not handle:
        return kernel32.GetLastError() == _WIN_ERROR_ACCESS_DENIED
    try:
        return kernel32.WaitForSingleObject(handle, 0) == _WIN_WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def _pid_running_posix(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:  # pragma: no cover - defensive
        return exc.errno != errno.ESRCH
    return True


def pid_running(pid: int) -> bool:
    """True if ``pid`` is a running process on this host."""
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        return _pid_running_windows(pid)
    return _pid_running_posix(pid)


def _read_lock(lock_path: Path) -> dict | None:
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_lock_exclusive(lock_path: Path) -> bool:
    """O_EXCL create + write owner payload. Return True on success, False if the file exists."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    payload = json.dumps(
        {"pid": os.getpid(), "created_utc": datetime.now(UTC).isoformat(), "hostname": socket.gethostname()},
        indent=2,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(payload)
    return True


def _is_stale(owner: dict | None, stale_hours: float) -> bool:
    if owner is None:
        return True
    host = owner.get("hostname")
    pid = owner.get("pid")
    if host == socket.gethostname() and isinstance(pid, int) and not pid_running(pid):
        return True
    created = owner.get("created_utc")
    if isinstance(created, str):
        try:
            age_h = (datetime.now(UTC) - datetime.fromisoformat(created)).total_seconds() / 3600.0
            if age_h > stale_hours:
                return True
        except ValueError:
            return True
    return False


def acquire_lock(lock_path: Path, stale_hours: float) -> None:
    """Acquire the lock at ``lock_path`` or raise ``LockHeldError``.

    A stale lock (dead owner PID, or age > ``stale_hours``) is reclaimed with a WARNING.
    """
    lock_path = Path(lock_path)
    if _write_lock_exclusive(lock_path):
        return

    owner = _read_lock(lock_path)
    if _is_stale(owner, stale_hours):
        logger.warning("Reclaiming stale lock %s (owner=%s)", lock_path, owner)
        lock_path.unlink(missing_ok=True)
        if _write_lock_exclusive(lock_path):
            return
        owner = _read_lock(lock_path)  # someone raced us; re-read for the error message

    contents = lock_path.read_text(encoding="utf-8") if lock_path.exists() else json.dumps(owner)
    message = (
        f"Lock held by a live process.\n"
        f"  lockfile: {lock_path}\n"
        f"  contents: {contents}\n"
        f"  To override: stop the owning process, then delete the lockfile:\n"
        f'    (Windows)  del "{lock_path}"\n'
        f'    (POSIX)    rm "{lock_path}"'
    )
    logger.error(message)
    raise LockHeldError(message)


def release_lock(lock_path: Path) -> None:
    """Release the lock only if this process (pid + hostname) owns it."""
    lock_path = Path(lock_path)
    owner = _read_lock(lock_path)
    if owner and owner.get("pid") == os.getpid() and owner.get("hostname") == socket.gethostname():
        lock_path.unlink(missing_ok=True)
