from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = _load("redact_check_site", ROOT / "tools" / "redact_check.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def test_html_files_are_scanned(scan: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    site = tmp_path / "site"
    site.mkdir()
    address = "person" + "@" + "example.org"
    (site / "index.html").write_text(f"<p>{address}</p>\n", encoding="utf-8")
    assert scan.main(["redact_check.py", str(site)]) == 1
    assert "index.html: email" in capsys.readouterr().out


def test_clean_html_passes(scan: ModuleType, tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<p>Whole-sample aggregates only.</p>\n", encoding="utf-8")
    assert scan.main(["redact_check.py", str(site)]) == 0


def test_missing_target_is_an_error_not_a_pass(
    scan: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert scan.main(["redact_check.py", str(tmp_path / "absent")]) == 2
    assert "target not found" in capsys.readouterr().out


def test_built_report_page_passes_the_scan(
    scan: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_report = _load("build_report_for_scan", ROOT / "docs" / "report" / "build_report.py")
    site = tmp_path / "site"
    index = build_report.build(site)
    assert index.is_file()
    code = scan.main(["redact_check.py", str(site)])
    assert code == 0, capsys.readouterr().out[-2000:]
