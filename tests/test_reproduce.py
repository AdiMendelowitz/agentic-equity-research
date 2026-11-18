"""The reproduction command compares artefacts correctly and can skip the slow control run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reproduce", ROOT / "docs" / "report" / "reproduce.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass under `from __future__ import annotations` looks the class's
    # module up in sys.modules (Python docs recipe for importing a source file directly).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


reproduce = _load()


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_identical_bytes_match(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.csv", b"x,y\n1,2\n")
    b = _write(tmp_path / "b.csv", b"x,y\n1,2\n")
    assert reproduce.compare_files(a, b) == "MATCH"


def test_one_changed_byte_differs(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.csv", b"x,y\n1,2\n")
    b = _write(tmp_path / "b.csv", b"x,y\n1,3\n")
    assert reproduce.compare_files(a, b) == "DIFFERS"


def test_missing_file_differs(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.csv", b"x\n")
    assert reproduce.compare_files(a, tmp_path / "absent.csv") == "DIFFERS"
    assert reproduce.compare_files(tmp_path / "absent.csv", a) == "DIFFERS"


def test_portable_treats_negative_zero_and_zero_alike() -> None:
    assert reproduce.portable(b"M -0 5.4 L -5.4 -0") == reproduce.portable(b"M 0 5.4 L -5.4 0")
    assert reproduce.portable(b"M -0.5 1") != reproduce.portable(b"M 0.5 1")
    assert reproduce.portable(b'id="mc91ad6a70d"') == reproduce.portable(b'id="m236070abc9"')


def test_svg_comparison_uses_the_portable_form(tmp_path: Path) -> None:
    fresh, committed = tmp_path / "fresh", tmp_path / "committed"
    fresh.mkdir()
    committed.mkdir()
    _write(fresh / "same.svg", b'<path id="mc91ad6a70d" d="M -0 1"/>')
    _write(committed / "same.svg", b'<path id="m236070abc9" d="M 0 1"/>')
    _write(fresh / "changed.svg", b'<path d="M 0 1"/>')
    _write(committed / "changed.svg", b'<path d="M 0 2"/>')
    _write(fresh / "only_fresh.svg", b"<svg/>")
    result = reproduce.compare_svg_dirs(fresh, committed)
    assert result == {"changed.svg": "DIFFERS", "only_fresh.svg": "DIFFERS", "same.svg": "MATCH"}


def test_skip_control_marks_it_skipped_without_running_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(scratch: Path) -> str:
        raise AssertionError("the control must not run with --skip-control")

    monkeypatch.setattr(reproduce, "SCRATCH", tmp_path / "reproduce")
    monkeypatch.setattr(reproduce, "run_control", explode)
    monkeypatch.setattr(reproduce, "check_figures", lambda scratch: {"one.svg": "MATCH"})
    monkeypatch.setattr(reproduce, "check_report", lambda scratch: "MATCH")
    assert reproduce.main(["--skip-control"]) == 0
    out = capsys.readouterr().out
    assert "SKIPPED" in out
    assert "detection_curve_v3.csv" in out
    assert out.strip().endswith("RESULT: REPRODUCED")
    assert not (tmp_path / "reproduce").exists()


def test_cards_quote_the_published_counts() -> None:
    model = (ROOT / "docs" / "cards" / "model-card.md").read_text(encoding="utf-8")
    data = (ROOT / "docs" / "cards" / "data-card.md").read_text(encoding="utf-8")
    assert "314 of 629" in model and "350 of 629" in model and "64 of the 120" in model
    assert "2,321 ticker-days" in data and "314 of 629" in data
    assert "22 completed with exit code 0" in data


def test_any_difference_gives_exit_code_one_and_keep_leaves_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(reproduce, "SCRATCH", tmp_path / "reproduce")
    monkeypatch.setattr(reproduce, "check_figures", lambda scratch: {"one.svg": "DIFFERS"})
    monkeypatch.setattr(reproduce, "check_report", lambda scratch: "MATCH")
    assert reproduce.main(["--skip-control", "--keep"]) == 1
    assert capsys.readouterr().out.strip().endswith("RESULT: DIFFERS")
    assert (tmp_path / "reproduce").is_dir()
