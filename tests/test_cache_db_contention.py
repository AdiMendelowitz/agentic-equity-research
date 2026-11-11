"""class 2 reproduction guard ("database is locked" x4 agents on 2026-07-12).

Replica measurement 2026-09-24: four agent threads plus a second OS process (the simulator's
shape) writing the same cache DB concurrently do NOT reproduce the lock error on the current tree,
where every connection is opened by src/utils/db.py with WAL and busy_timeout=30000. The
2026-07-12 mechanism itself stays unmeasured [?]: that run predates the centralisation, and
any claim about it must be checked against the code as of that date, not HEAD. This test
keeps the present-day guarantee from regressing; the restart runs with the audit watching.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRITES = 400


def _writer(db: str, tag: str, errors: list) -> None:
    from aer.data.cache import Cache
    from aer.utils.cache import _DomainCache

    c = _DomainCache(Cache(db_path=db), 60, "data")
    try:
        for i in range(WRITES):
            c.set(f"{tag}:{i}", {"v": i}, 60)
            c.get(f"{tag}:{i}")
    except Exception as e:  # pragma: no cover - the failure being guarded
        errors.append(f"{tag}: {type(e).__name__}: {e}")


def test_concurrent_threads_and_process_do_not_lock(tmp_path):
    db = str(tmp_path / "contention.db")
    from aer.data.cache import Cache

    Cache(db_path=db)  # schema once
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, %r)\n"
            "from aer.data.cache import Cache\n"
            "from aer.utils.cache import _DomainCache\n"
            "c = _DomainCache(Cache(db_path=%r), 60, 'data')\n"
            "[c.set('proc:%%d' %% i, {'v': i}, 60) for i in range(%d)]\n" % (str(ROOT), db, WRITES),
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    errors: list[str] = []
    threads = [threading.Thread(target=_writer, args=(db, f"agent{n}", errors)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    out, err = child.communicate(timeout=120)
    assert errors == []
    assert child.returncode == 0, err[-2000:]
    from aer.utils.cache import _DomainCache

    c = _DomainCache(Cache(db_path=db), 60, "data")
    assert all(c.get(f"agent{n}:{WRITES - 1}") == {"v": WRITES - 1} for n in range(4))
    assert c.get(f"proc:{WRITES - 1}") == {"v": WRITES - 1}
