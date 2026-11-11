"""Redaction scan for the public tree. Exit 1 on any hit. Runs in pre-commit and CI.

Rules: secrets, absolute Windows paths, internal process ids, planning labels, predecessor names (matched by hash),
social usernames, emails, long quoted passages, price-level sentences in replay fixtures, PNG metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", "notes", "__pycache__", ".cache", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
SKIP_FILES = {"CLAUDE.md", ".env"}  # gitignored internal files; never part of the public tree
PROSE_EXT = {
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".csv",
}  # rules that make no sense in code (handles, quotations) apply here only
TEXT_EXT = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".jsonl", ".csv", ".txt", ".cfg", ".ini"}

RULES: dict[str, re.Pattern[str]] = {
    "secret": re.compile(
        r"\b(sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|gsk_[A-Za-z0-9]{16,}|nvapi-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,})"
    ),
    "windows_path": re.compile(r"[A-Za-z]:\\\\?Users\\\\?", re.I),
    "process_id": re.compile(r"\b(F-\d{2,3}|A\d{2}|D-\d{1,2}|I-\d{1,2}|R-\d|CR-\d{2}|LEDGER:\d+|PHASE [0-9A-H]\b)"),
    # planning labels of the private predecessor (phase, finding and review numbers, context-file names)
    "planning_label": re.compile(
        r"\b(Phase \d+[a-z]?\b|FINDING-\d+|Finding \d+\b(?!\s?%)|review (fix )?[A-Z]-\d+|review #\d+|\d+-CONTEXT\.md"
        r"|RESEARCH A\d|master_remediation|PROF-\d+|Wave \d\b|Pitfall \d+|hardening item \d+)"
    ),
    # review and plan ids the first label rule missed (found by the 2026-11-11 read-through)
    "review_id": re.compile(
        r"\b(PF-\d{2}|REQ-\d{1,3}|BL-\d{2}|IN-\d{2}|SC-\d|QW-[A-F]|WI-\d|B-ESC-\d|DATA-\d{2}|RISK-\d{2}|FACTOR-\d{2}"
        r"|[BHLM]\d{1,2}[ab]? fix|3[0-2][ab]?-0\d|M-\d{1,2}(?!\d))\b|_phase\d+|quickwins"
    ),
    "social_handle": re.compile(r"(?<![\w/])(u/|@)[A-Za-z0-9_]{3,}\b"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}"),
    "long_quote": re.compile(r"[\"“][^\"”\n]{120,}[\"”]"),
}
# Names of the private predecessor project are matched by hash so this file does not carry them in clear.
DENY_HASHES = {
    "01259da07d6c6038ea4bb7271f14145d5b1de29b3b0aeaa5c7c9b5bd2e14be92",
    "22555a6db4ba1a341ebe667952237dfc6900d3d975847c22c88ae7164b7d687f",
    "2f81dce25ea05d9bfaf3272239756414ffb88084ac1f61bcc57faf348ab72873",
    "7b466973134cd78604fb185cc981d72e339cf67ed9e1c941f10e1a9b10406fa1",
    "87325fb779505a1e7b923dd927dcf352499b59ed9ba5dddd9d6278c40c8c0264",
    "9aeab24f41872a992858389cab389af015dd4da904c6accb9ed43161b77b0253",
    "a5b30aef7131f5f7e9467a6a25da7d64a9582e0d4d0c4b34469f4a89e9923a1f",
    "aaef3928cdb8052e107b1f5fb01b60a1e9461a72fbd484f0df71b8553196d0e9",
    "d7b121c3bf331c79c003a1837bb51c97e4edfa2fb402b3b5f80776186835277f",
}
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")
PRICE_LEVEL = re.compile(
    r"\$\s?\d"
)  # replay fixtures carry no currency amount at all (ADR-05); levels are redacted at extraction
ALLOW = {"email": {"noreply@github.com", "name@example.com"}}


def scan_text(path: Path, text: str, hits: list[str]) -> None:
    if path.resolve() == Path(__file__).resolve():
        return
    for m in RULES["review_id"].finditer(path.name):
        hits.append(f"{path.relative_to(ROOT)}: file_name: {m.group(0)!r}")
    for tok in set(TOKEN.findall(text)):
        if hashlib.sha256(tok.lower().encode()).hexdigest() in DENY_HASHES:
            hits.append(f"{path.relative_to(ROOT)}: denied_name: {tok!r}")
    for name, rx in RULES.items():
        for m in rx.finditer(text):
            if m.group(0) in ALLOW.get(name, set()):
                continue
            if name in {"long_quote", "social_handle"} and path.suffix not in PROSE_EXT:
                continue  # prose-only rules; fixture text is scanned separately below
            if name == "long_quote" and path.suffix in {".json", ".jsonl", ".csv"}:
                continue
            hits.append(f"{path.relative_to(ROOT)}: {name}: {m.group(0)[:60]!r}")
    if path.is_relative_to(ROOT / "src/aer/demo/fixtures") and path.suffix == ".json":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            hits.append(f"{path.relative_to(ROOT)}: invalid_json: {exc}")
            return
        for field, value in _walk_strings(obj):
            for m in PRICE_LEVEL.finditer(value):
                hits.append(f"{path.relative_to(ROOT)}: price_level[{field}]: {m.group(0)[:60]!r}")
            for m in RULES["long_quote"].finditer(value):
                hits.append(f"{path.relative_to(ROOT)}: long_quote[{field}]: {m.group(0)[:60]!r}")


def _walk_strings(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{prefix}[{i}]")
    elif isinstance(obj, str):
        yield prefix, obj


def scan_png(path: Path, hits: list[str]) -> None:
    data = path.read_bytes()
    for chunk in (b"tEXt", b"iTXt", b"zTXt", b"eXIf"):
        if chunk in data:
            hits.append(f"{path.relative_to(ROOT)}: png_metadata: {chunk.decode()}")


def main(argv: list[str]) -> int:
    targets = [Path(a).resolve() for a in argv[1:]] or [ROOT]
    hits: list[str] = []
    for t in targets:
        files = (
            [t]
            if t.is_file()
            else [
                p
                for p in t.rglob("*")
                if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS) and p.name not in SKIP_FILES
            ]
        )
        for p in files:
            if p.suffix == ".png":
                scan_png(p, hits)
            elif p.suffix in TEXT_EXT:
                scan_text(p, p.read_text(encoding="utf-8", errors="replace"), hits)
    for h in hits:
        print(h)
    print(f"redact_check: {len(hits)} hit(s) in {len(targets)} target(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
