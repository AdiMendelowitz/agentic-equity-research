"""Tests for src/utils/locking.py — single-instance PID lockfile (B2b)."""

from __future__ import annotations

import json
import os
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aer.utils.locking import (
    LockHeldError,
    acquire_lock,
    pid_running,
    release_lock,
)


def test_pid_running_true_for_self() -> None:
    assert pid_running(os.getpid()) is True


def test_pid_running_false_for_nonpositive() -> None:
    assert pid_running(0) is False
    assert pid_running(-1) is False


def test_pid_running_false_for_dead_pid() -> None:
    # A very high PID is almost certainly not a live process.
    assert pid_running(2_000_000_000) is False


def test_acquire_creates_lockfile_with_owner(tmp_path: Path) -> None:
    lock = tmp_path / "pipeline.lock"
    acquire_lock(lock, stale_hours=6)
    assert lock.exists()
    owner = json.loads(lock.read_text(encoding="utf-8"))
    assert owner["pid"] == os.getpid()
    assert owner["hostname"] == socket.gethostname()
    assert "created_utc" in owner


def test_second_acquire_by_live_owner_raises(tmp_path: Path) -> None:
    lock = tmp_path / "pipeline.lock"
    acquire_lock(lock, stale_hours=6)
    with pytest.raises(LockHeldError) as exc:
        acquire_lock(lock, stale_hours=6)
    # Error message names the lockfile path and manual-removal instructions.
    msg = str(exc.value)
    assert str(lock) in msg
    assert "delete the lockfile" in msg


def test_release_removes_own_lock(tmp_path: Path) -> None:
    lock = tmp_path / "pipeline.lock"
    acquire_lock(lock, stale_hours=6)
    release_lock(lock)
    assert not lock.exists()
    # Re-acquire succeeds after release.
    acquire_lock(lock, stale_hours=6)
    assert lock.exists()


def test_stale_lock_by_dead_pid_is_reclaimed(tmp_path: Path) -> None:
    lock = tmp_path / "pipeline.lock"
    # Write a lock owned by a dead PID on THIS host so liveness check fires.
    lock.write_text(
        json.dumps(
            {
                "pid": 2_000_000_000,
                "created_utc": datetime.now(UTC).isoformat(),
                "hostname": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )
    acquire_lock(lock, stale_hours=6)  # dead owner -> reclaimed, no raise
    owner = json.loads(lock.read_text(encoding="utf-8"))
    assert owner["pid"] == os.getpid()


def test_stale_lock_by_age_is_reclaimed(tmp_path: Path) -> None:
    lock = tmp_path / "pipeline.lock"
    # Live PID (ourselves) but aged past stale_hours -> reclaimed via age bound.
    old = datetime.now(UTC) - timedelta(hours=9)
    lock.write_text(
        json.dumps({"pid": os.getpid(), "created_utc": old.isoformat(), "hostname": socket.gethostname()}),
        encoding="utf-8",
    )
    acquire_lock(lock, stale_hours=6)  # age > 6h -> reclaimed
    assert lock.exists()


def test_release_does_not_remove_foreign_lock(tmp_path: Path) -> None:
    lock = tmp_path / "pipeline.lock"
    lock.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "created_utc": datetime.now(UTC).isoformat(),
                "hostname": "some-other-host",
            }
        ),
        encoding="utf-8",
    )
    release_lock(lock)  # hostname mismatch -> must NOT delete
    assert lock.exists()


def test_corrupt_lockfile_is_treated_as_stale(tmp_path: Path) -> None:
    lock = tmp_path / "pipeline.lock"
    lock.write_text("{ this is not json", encoding="utf-8")
    acquire_lock(lock, stale_hours=6)  # unreadable owner -> stale -> reclaimed
    owner = json.loads(lock.read_text(encoding="utf-8"))
    assert owner["pid"] == os.getpid()
