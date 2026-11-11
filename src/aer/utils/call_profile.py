"""Wall-time profile of the pipeline graph, per node and per LLM stage, MASTER_PLAN task 2).

A development instrument: production code never imports it. ``scripts/dev/profile_run.py``
wraps callables at runtime with:func:`timed`, runs ``invoke_pipeline`` for a few tickers in an
isolated sandbox, and writes:func:`summarise` as Markdown. Nothing here changes a return value
or swallows an exception; a wrapped call behaves exactly as the original.

Attribution: an LLM stage is charged to the graph node found on the calling thread's stack
(``_run_fundamental`` -> ``fundamental``). LangGraph runs the five analyst nodes in parallel,
so node wall times overlap and their sum can exceed the ticker's wall time; the ticker row is the
only figure that adds up to elapsed time.
"""

from __future__ import annotations

import functools
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

NODE_FUNCTIONS: dict[str, str] = {
    "_run_fundamental": "fundamental",
    "_run_technical": "technical",
    "_run_macro": "macro",
    "_run_sentiment": "sentiment",
    "_run_positioning": "positioning",
    "_run_news": "news",
    "_synthesize": "synthesize",
    "_run_backtesting": "backtesting",
    "_run_options_strategy": "options_strategy",
    "_post_process": "post_process",
    "_run_ml_prediction": "ml_prediction",
    "_run_professor_critique": "professor_critique",
}


@dataclass(frozen=True)
class Event:
    kind: str  # "ticker" | "node" | "llm"
    name: str  # ticker symbol, node name, or LLM stage label
    node: str | None  # graph node an LLM stage ran under (None outside any node)
    ticker: str | None
    elapsed_s: float
    ok: bool


class Recorder:
    """Thread-safe event sink. ``ticker`` is set by the driver before each ticker runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[Event] = []
        self.ticker: str | None = None

    def add(self, event: Event) -> None:
        with self._lock:
            self.events.append(event)


def attribute_node(frame: FrameType | None, functions: dict[str, str] = NODE_FUNCTIONS) -> str | None:
    """Walk ``frame`` outwards and return the first graph node whose function is on the stack."""
    while frame is not None:
        node = functions.get(frame.f_code.co_name)
        if node is not None:
            return node
        frame = frame.f_back
    return None


def timed(
    fn: Callable[..., Any],
    recorder: Recorder,
    kind: str,
    label: Callable[[tuple, dict], str] | str,
    ok: Callable[[Any], bool] = lambda _out: True,
) -> Callable[..., Any]:
    """Return ``fn`` wrapped so every call records one :class:`Event`, including a call that raises."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        name = label(args, kwargs) if callable(label) else label
        node = attribute_node(sys._getframe(1)) if kind == "llm" else name
        ticker = recorder.ticker  # read at call start: a worker thread may finish after the next ticker began
        start = time.perf_counter()
        succeeded = False
        try:
            out = fn(*args, **kwargs)
            succeeded = bool(ok(out))
            return out
        finally:
            recorder.add(Event(kind, name, node, ticker, time.perf_counter() - start, succeeded))

    return wrapper


def summarise(events: Iterable[Event]) -> dict[str, list[dict[str, Any]]]:
    """Aggregate events into three tables: tickers, nodes, and LLM stages per node."""
    events = list(events)
    tickers = [{"ticker": e.name, "wall_s": round(e.elapsed_s, 1), "ok": e.ok} for e in events if e.kind == "ticker"]
    n_tickers = max(len(tickers), 1)

    nodes: dict[str, dict[str, Any]] = {}
    for e in events:
        if e.kind == "node":
            row = nodes.setdefault(
                e.name, {"node": e.name, "wall_s": 0.0, "llm_s": 0.0, "llm_calls": 0, "llm_failed": 0}
            )
            row["wall_s"] += e.elapsed_s
    stages: dict[tuple[str, str], dict[str, Any]] = {}
    for e in events:
        if e.kind != "llm":
            continue
        node = e.node or "(outside graph)"
        row = nodes.setdefault(node, {"node": node, "wall_s": 0.0, "llm_s": 0.0, "llm_calls": 0, "llm_failed": 0})
        row["llm_s"] += e.elapsed_s
        row["llm_calls"] += 1
        row["llm_failed"] += 0 if e.ok else 1
        st = stages.setdefault((node, e.name), {"node": node, "stage": e.name, "calls": 0, "failed": 0, "total_s": 0.0})
        st["calls"] += 1
        st["failed"] += 0 if e.ok else 1
        st["total_s"] += e.elapsed_s

    node_rows = []
    for row in sorted(nodes.values(), key=lambda r: -max(r["wall_s"], r["llm_s"])):
        node_rows.append(
            {
                **row,
                "wall_s": round(row["wall_s"], 1),
                "llm_s": round(row["llm_s"], 1),
                "non_llm_s": round(max(row["wall_s"] - row["llm_s"], 0.0), 1),
                "wall_s_per_ticker": round(row["wall_s"] / n_tickers, 1),
            }
        )
    stage_rows = [
        {**st, "total_s": round(st["total_s"], 1), "mean_s": round(st["total_s"] / st["calls"], 2)}
        for st in sorted(stages.values(), key=lambda s: -s["total_s"])
    ]
    return {"tickers": tickers, "nodes": node_rows, "stages": stage_rows}


def to_markdown(summary: dict[str, list[dict[str, Any]]]) -> str:
    """Render :func:`summarise` output as three Markdown tables."""

    def table(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "(none)\n"
        cols = list(rows[0])
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
        return "\n".join(lines) + "\n"

    return (
        "## Tickers (elapsed wall time; the only rows that add up)\n\n"
        + table(summary["tickers"])
        + "\n## Nodes (analyst nodes run in parallel, so node walls overlap)\n\n"
        + table(summary["nodes"])
        + "\n## LLM stages per node (a failed stage is time spent before the cascade moved on)\n\n"
        + table(summary["stages"])
    )


def snapshot(root: Path, skip_dirs: frozenset[str]) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) of every file under ``root``, skipping any directory named in ``skip_dirs``."""
    out: dict[str, tuple[int, int]] = {}
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for p in entries:
            if p.is_dir() and not p.is_symlink():
                if p.name not in skip_dirs:
                    stack.append(p)
            elif p.is_file():
                st = p.stat()
                out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns)
    return out


def changed_files(before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]) -> list[str]:
    """Paths created, removed or modified between two :func:`snapshot` results."""
    return sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
