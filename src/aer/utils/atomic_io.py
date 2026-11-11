"""Atomic file writes: temp file in the destination directory, fsync, then ``os.replace``.

``os.replace`` is atomic on POSIX and Windows when source and destination share a filesystem,
which is guaranteed here because the temp file is created in the destination's own directory.
This prevents a truncated/empty file becoming visible on a mid-write crash and avoids the
Windows ``FileExistsError`` that a plain ``rename`` raises over an existing file.

Reference: Python docs, ``os.replace``. https://docs.python.org/3/library/os.html#os.replace
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

_ENCODING = "utf-8"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path`` (parent dirs created as needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str, *, encoding: str = _ENCODING) -> None:
    """Atomically write ``text`` to ``path``."""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2, sort_keys: bool = False) -> None:
    """Atomically serialize ``payload`` to JSON and write it to ``path``.

    ``sort_keys`` defaults to False so caller-controlled key order (``schema_version`` first)
    is preserved.
    """
    atomic_write_text(path, json.dumps(payload, indent=indent, sort_keys=sort_keys, default=str))


def read_json(path: Path) -> Any:
    """Read and parse JSON from ``path`` (raises on absent/malformed)."""
    return json.loads(Path(path).read_text(encoding=_ENCODING))
