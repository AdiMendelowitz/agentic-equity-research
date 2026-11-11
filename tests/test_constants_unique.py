"""Upper-case constants are defined once: not twice in a module, not differently across modules."""

import ast
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "aer"


def _constant_modules() -> list[Path]:
    modules = set(PKG.rglob("*constants*.py"))
    modules.add(PKG / "utils" / "cost_thresholds.py")
    return sorted(modules)


def _bindings(source: str) -> list[tuple[str, str]]:
    """Return (name, unparsed value) for each module-level upper-case binding, in source order."""
    found: list[tuple[str, str]] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                found.append((target.id, ast.unparse(value)))
    return found


def _duplicates_within(source: str) -> list[str]:
    seen: set[str] = set()
    duplicated: list[str] = []
    for name, _ in _bindings(source):
        if name in seen:
            duplicated.append(name)
        seen.add(name)
    return duplicated


def test_no_constant_bound_twice_in_one_module() -> None:
    modules = _constant_modules()
    assert modules
    offenders = {
        str(path.relative_to(ROOT)): dups
        for path in modules
        if (dups := _duplicates_within(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}


def test_no_constant_has_conflicting_values_across_modules() -> None:
    values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path in _constant_modules():
        for name, value in _bindings(path.read_text(encoding="utf-8")):
            values[name][value].add(str(path.relative_to(ROOT)))
    conflicts = {
        name: {v: sorted(p) for v, p in by_value.items()} for name, by_value in values.items() if len(by_value) > 1
    }
    assert conflicts == {}


def test_helper_detects_a_duplicate() -> None:
    assert _duplicates_within("LIMIT = 1\nOTHER: int = 2\nLIMIT = 3\n") == ["LIMIT"]
    assert _duplicates_within("LIMIT = 1\nOTHER = 2\n") == []
