"""real_key_cache must reach every module that bound a cache factory at import time.

Before the fix the fixture patched aer.utils.cache plus a hand list of ~14 consumers, while ~40
modules do `from aer.utils.cache import <factory>`; the rest kept calling the REAL factory, so a
'contract' test through them read and wrote the real cache DB. This test imports every such
module, then asserts each bound factory returns the fixture's tmp cache."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "aer"
_IMPORT_RE = re.compile(r"^from aer\.utils\.cache import (.+)$", re.M)
FACTORIES = ("data_cache", "agent_cache", "brief_cache", "news_cache", "lseg_cache")


def _consumers():
    out = []
    for p in sorted(PKG.rglob("*.py")):
        if p.name == "cache.py" and p.parent.name == "utils":
            continue
        m = _IMPORT_RE.findall(p.read_text(encoding="utf-8"))
        names = {n.strip(" ()") for line in m for n in line.split(",")}
        hit = sorted(names & set(FACTORIES))
        if hit and "pages" not in p.parts:
            mod = ".".join(p.relative_to(SRC).with_suffix("").parts)
            out.append((mod, hit))
    return out


CONSUMERS = _consumers()


@pytest.fixture(scope="module", autouse=True)
def _import_all_consumers():
    for mod, _ in CONSUMERS:
        importlib.import_module(mod)


def test_every_bound_factory_returns_the_fixture_cache(real_key_cache):
    import aer.utils.cache as uc

    assert len(CONSUMERS) >= 18
    missed = []
    for mod, names in CONSUMERS:
        m = importlib.import_module(mod)
        for n in names:
            if getattr(m, n)() is not getattr(uc, n)():
                missed.append(f"{mod}.{n}")
    assert missed == []
