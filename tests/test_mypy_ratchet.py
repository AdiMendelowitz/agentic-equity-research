from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mypy_ratchet", ROOT / "tools" / "mypy_ratchet.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ratchet = _load()


def test_count_is_read_from_the_summary_line() -> None:
    output = "src/aer/x.py:1: error: something  [misc]\nFound 12 errors in 3 files (checked 103 source files)\n"
    assert ratchet.error_count(output) == 12
    assert ratchet.error_count("Found 1 error in 1 file (checked 2 source files)\n") == 1


def test_clean_run_counts_zero_and_missing_summary_is_unknown() -> None:
    assert ratchet.error_count("Success: no issues found in 103 source files\n") == 0
    assert ratchet.error_count("mypy: can't read file\n") is None


def test_verdict_fails_only_above_the_ceiling_or_when_unknown() -> None:
    assert ratchet.verdict(11, ceiling=10)[0] == 1
    assert ratchet.verdict(10, ceiling=10)[0] == 0
    assert ratchet.verdict(9, ceiling=10)[0] == 0
    assert "lower MAX_ERRORS to 9" in ratchet.verdict(9, ceiling=10)[1]
    assert ratchet.verdict(None, ceiling=10)[0] == 2


def test_ci_and_the_local_check_run_the_ratchet_as_a_blocking_step() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "uv run python tools/mypy_ratchet.py" in ci
    assert "mypy || true" not in ci
    assert "continue-on-error" not in ci
    check = (ROOT / "tools" / "check.ps1").read_text(encoding="utf-8")
    assert 'Invoke-Step "mypy" "uv run python tools/mypy_ratchet.py" $true' in check
