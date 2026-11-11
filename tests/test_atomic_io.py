"""Tests for src/utils/atomic_io.py — atomic write helpers (B2b)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aer.utils.atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    read_json,
)


def test_write_json_roundtrips(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    payload = {"b": 2, "a": 1, "nested": {"x": [1, 2, 3]}}
    atomic_write_json(target, payload)
    assert read_json(target) == payload


def test_json_key_order_preserved_by_default(tmp_path: Path) -> None:
    # sort_keys defaults False so schema_version-first ordering survives.
    target = tmp_path / "ordered.json"
    atomic_write_json(target, {"schema_version": 1, "ticker": "AAPL"})
    assert list(json.loads(target.read_text(encoding="utf-8")).keys()) == ["schema_version", "ticker"]


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c" / "deep.json"
    atomic_write_json(target, {"ok": True})
    assert target.exists()
    assert read_json(target) == {"ok": True}


def test_write_json_default_str_serializes_dates(tmp_path: Path) -> None:
    from datetime import date

    target = tmp_path / "dated.json"
    atomic_write_json(target, {"d": date(2026, 7, 8)})
    assert read_json(target) == {"d": "2026-07-08"}


def test_overwrite_replaces_atomically(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"v": 1})
    atomic_write_json(target, {"v": 2})
    assert read_json(target) == {"v": 2}


def test_no_temp_files_left_behind(tmp_path: Path) -> None:
    target = tmp_path / "clean.json"
    atomic_write_json(target, {"v": 1})
    atomic_write_json(target, {"v": 2})
    # The atomic temp files are named ".<name>.*.tmp"; assert none leaked.
    # (Other dirs like conftest's logs/ are irrelevant to this test's intent.)
    tmp_leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert tmp_leftovers == [], f"unexpected temp files: {tmp_leftovers}"


def test_write_text_and_bytes(tmp_path: Path) -> None:
    t = tmp_path / "t.txt"
    atomic_write_text(t, "héllo")
    assert t.read_text(encoding="utf-8") == "héllo"

    b = tmp_path / "b.bin"
    atomic_write_bytes(b, b"\x00\x01\x02")
    assert b.read_bytes() == b"\x00\x01\x02"


def test_temp_cleaned_up_when_replace_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force os.replace to fail AFTER the temp file has been created and written,
    # so the except-branch cleanup is what's under test. Target must not appear
    # and no .tmp leftover must remain.
    import aer.utils.atomic_io as aio

    target = tmp_path / "fail.json"

    def _boom(src: object, dst: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(aio.os, "replace", _boom)

    with pytest.raises(OSError):
        atomic_write_json(target, {"v": 1})

    assert not target.exists()
    tmp_leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert tmp_leftovers == [], f"temp file left after failure: {tmp_leftovers}"


def test_read_json_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json(tmp_path / "nope.json")
