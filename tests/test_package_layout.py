"""Every source package under src/aer is a regular package, never an implicit namespace package."""

from __future__ import annotations

from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "aer"


def test_package_root_exists() -> None:
    assert (PKG / "__init__.py").is_file()


def test_no_implicit_namespace_packages_under_src() -> None:
    missing = sorted(
        str(d.relative_to(PKG))
        for d in PKG.rglob("*")
        if d.is_dir() and d.name != "__pycache__" and any(d.glob("*.py")) and not (d / "__init__.py").exists()
    )
    assert missing == [], f"directories with .py files but no __init__.py: {missing}"
