"""Importing every aer module must be hermetic: no network, no database files, no credentials."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIPPED_PARTS = {".venv", "node_modules", ".git"}
EXPLICIT_KEYS = {"ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY", "NVIDIA_NIM_API_KEY"}

SCRIPT = """
import importlib
import json
import pkgutil
import socket

connects = []


def _refuse(self, address):
    connects.append(repr(address))
    raise ConnectionRefusedError("network access is blocked in the hermetic test")


socket.socket.connect = _refuse
socket.socket.connect_ex = _refuse

import aer

imported = 0
failed = []
for info in pkgutil.walk_packages(aer.__path__, "aer."):
    try:
        importlib.import_module(info.name)
        imported += 1
    except BaseException as exc:
        failed.append((info.name, repr(exc)))
print(json.dumps({"imported": imported, "failed": failed, "connects": connects}))
"""


def _db_files() -> set[Path]:
    return {p for p in ROOT.rglob("*.db") if not (set(p.relative_to(ROOT).parts) & SKIPPED_PARTS)}


def _clean_env(home: Path) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.endswith("_API_KEY") or k.endswith("_TOKEN") or k.startswith("AER_") or k in EXPLICIT_KEYS)
    }
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    return env


def test_importing_every_module_is_hermetic(tmp_path: Path) -> None:
    dbs_before = _db_files()
    cache_before = (ROOT / ".cache").exists()

    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        cwd=tmp_path,
        env=_clean_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])

    assert report["failed"] == []
    assert report["connects"] == []
    assert report["imported"] >= 60
    assert _db_files() == dbs_before
    assert (ROOT / ".cache").exists() == cache_before
    assert list(tmp_path.rglob("*.db")) == []
