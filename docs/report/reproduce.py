"""Rebuild the published artefacts into a scratch folder and compare them with the committed ones.

Usage: uv run python docs/report/reproduce.py [--skip-control] [--keep]

Three steps, one output line per artefact (MATCH, DIFFERS or SKIPPED):

1. Control v3: the pre-registered control is run again with the functions docs/report/run_control_v3.py calls,
   written to the scratch folder and compared byte for byte with docs/data/detection_curve_v3.csv.
2. Figures: docs/report/build_figures.py is run with --out <scratch>/img and each SVG is compared with docs/img,
   ignoring only the sign of zero and the content-hashed element ids.
3. Report: docs/report/build_report.py builds the page into <scratch>/site; success is reported as MATCH.

The scratch folder is .scratch/reproduce inside the repository (gitignored). It is created and removed by the
script unless --keep is given. Nothing is written outside the repository folder, and docs/data is never written.
The last line is `RESULT: REPRODUCED`, or `RESULT: DIFFERS` with exit code 1 when any artefact differs.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / ".scratch" / "reproduce"
REPORT_DIR = ROOT / "docs" / "report"
COMMITTED_CONTROL = ROOT / "docs" / "data" / "detection_curve_v3.csv"
COMMITTED_IMG = ROOT / "docs" / "img"

MATCH = "MATCH"
DIFFERS = "DIFFERS"
SKIPPED = "SKIPPED"


def portable(svg: bytes) -> bytes:
    """Drop what differs between platforms without changing the drawing: the sign of zero and hashed ids.

    Copied from `_portable` in tests/test_report_figures.py so that this script and the test agree.
    """
    text = re.sub(rb"(?<![\d.])-0(?![\d.])", b"0", svg)
    return re.sub(rb"\b([mp])[0-9a-f]{10}\b", rb"\1ID", text)


def compare_files(fresh: Path, committed: Path, normalise: Callable[[bytes], bytes] | None = None) -> str:
    """Return MATCH when both files exist and hold the same bytes (after `normalise`), else DIFFERS."""
    if not fresh.is_file() or not committed.is_file():
        return DIFFERS
    left, right = fresh.read_bytes(), committed.read_bytes()
    if normalise is not None:
        left, right = normalise(left), normalise(right)
    return MATCH if left == right else DIFFERS


def compare_svg_dirs(fresh_dir: Path, committed_dir: Path) -> dict[str, str]:
    """Compare every SVG name found in either folder; a name missing on one side is DIFFERS."""
    names = sorted({p.name for p in fresh_dir.glob("*.svg")} | {p.name for p in committed_dir.glob("*.svg")})
    return {name: compare_files(fresh_dir / name, committed_dir / name, portable) for name in names}


def load_module(name: str, path: Path) -> ModuleType:
    """Load a source file as a module, registering it in sys.modules first (dataclasses need that)."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_control(scratch: Path) -> str:
    """Re-run the pre-registered control v3 into `scratch` and compare it with the committed CSV.

    Calls the functions docs/report/run_control_v3.py calls: prereg_v3.seeds and SIGNALS, run_cell_v3 and
    write_csv_v3. Slow: it fits every cell of the full grid.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from aer.demo import prereg_v3
    from aer.demo.control_v3 import CellV3, run_cell_v3, write_csv_v3

    started = time.perf_counter()
    cells: list[CellV3] = []
    for signal in prereg_v3.SIGNALS:
        for seed in prereg_v3.seeds():
            cells.append(run_cell_v3(signal, seed))
    out = scratch / "detection_curve_v3.csv"
    scratch.mkdir(parents=True, exist_ok=True)
    write_csv_v3(cells, out)
    print(f"control v3: {len(cells)} cells in {time.perf_counter() - started:.1f} seconds")
    return compare_files(out, COMMITTED_CONTROL)


def check_figures(scratch: Path) -> dict[str, str]:
    """Rebuild the figures into `<scratch>/img` with build_figures.main and compare with docs/img."""
    module = load_module("build_figures", REPORT_DIR / "build_figures.py")
    out = scratch / "img"
    status = module.main(["--out", str(out)])
    if status != 0:
        return {"build_figures.py": DIFFERS}
    return compare_svg_dirs(out, COMMITTED_IMG)


def check_report(scratch: Path) -> str:
    """Build the report page into `<scratch>/site` with build_report.build and report success."""
    module = load_module("build_report", REPORT_DIR / "build_report.py")
    index = Path(module.build(scratch / "site"))
    return MATCH if index.is_file() and index.stat().st_size > 0 else DIFFERS


def _guarded(step: Callable[[], str], label: str) -> str:
    """Run a step that returns one status; an exception is reported on stderr and counts as DIFFERS."""
    try:
        return step()
    except Exception as error:  # any failure to rebuild means the artefact is not reproduced
        print(f"{label}: failed with {type(error).__name__}: {error}", file=sys.stderr)
        return DIFFERS


def _remove_scratch(scratch: Path) -> None:
    """Remove the scratch folder; refuses any folder not named `reproduce`."""
    if scratch.name == "reproduce" and scratch.exists():
        shutil.rmtree(scratch)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the three steps, print one line per artefact and return 0 when nothing differs, else 1."""
    parser = argparse.ArgumentParser(
        description="Rebuild the published artefacts into .scratch/reproduce and compare them with the committed ones.",
        epilog="The control step is slow: it re-runs every cell of the pre-registered grid (all signals and seeds).",
    )
    parser.add_argument(
        "--skip-control",
        action="store_true",
        help="skip the control v3 re-run (slow: it fits every cell of the pre-registered grid)",
    )
    parser.add_argument("--keep", action="store_true", help="keep the scratch folder .scratch/reproduce")
    args = parser.parse_args(argv)

    scratch = SCRATCH
    _remove_scratch(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str]] = []
    try:
        if args.skip_control:
            results.append(("docs/data/detection_curve_v3.csv", SKIPPED))
        else:
            results.append(("docs/data/detection_curve_v3.csv", _guarded(lambda: run_control(scratch), "control v3")))
        try:
            figures = check_figures(scratch)
        except Exception as error:
            print(f"figures: failed with {type(error).__name__}: {error}", file=sys.stderr)
            figures = {"docs/img": DIFFERS}
        results.extend((f"docs/img/{name}", status) for name, status in figures.items())
        results.append(("report page", _guarded(lambda: check_report(scratch), "report")))
    finally:
        if not args.keep:
            _remove_scratch(scratch)
    for label, status in results:
        print(f"{status:8} {label}")
    reproduced = all(status != DIFFERS for _, status in results)
    print("RESULT: REPRODUCED" if reproduced else "RESULT: DIFFERS")
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
