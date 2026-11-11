"""(Adi 2026-09-29): the scheduled pipeline writes no opinions on a day the NYSE is closed.

The task trigger stays daily, so run_status.json is refreshed every day and the 30 h dead-man's switch
(PIPELINE_HEARTBEAT_STALE, logon_check) keeps its meaning. A closed day exits 0 with an empty per_ticker map.
PIPELINE_RUN_NON_TRADING_DAYS=1 overrides it for a manual run.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import aer.graph as graph_mod
import aer.run_pipeline as rp
import aer.utils.watchlist as watchlist_mod
from aer.models import InvestmentBrief
from aer.utils.artifact_schema import RUN_STATUS_FILENAME, load_run_status
from aer.utils.pipeline_heartbeat import HEARTBEAT_FILENAME

_ET = ZoneInfo("America/New_York")
TRADING_DAY = datetime(2026, 7, 7, 1, 30, tzinfo=_ET)  # Tuesday
SATURDAY = datetime(2026, 7, 11, 1, 30, tzinfo=_ET)
HOLIDAY = datetime(2026, 7, 3, 1, 30, tzinfo=_ET)  # Friday, Independence Day observed


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AER_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("AER_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("PIPELINE_RUN_NON_TRADING_DAYS", raising=False)
    monkeypatch.setattr(rp, "_git_sha", lambda: "3ad3b29a9f00c1d2")
    monkeypatch.setattr(watchlist_mod, "load_watchlist", lambda: ["AAPL"])
    return tmp_path


@pytest.fixture
def invoked(monkeypatch):
    calls: list[str] = []

    def fake(ticker: str):
        calls.append(ticker)
        return InvestmentBrief(ticker=ticker, company_name="A", recommendation="BUY", conviction=0.8), {
            "professor_critique": "c"
        }

    monkeypatch.setattr(graph_mod, "invoke_pipeline", fake)
    return calls


def _artifacts(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.mark.parametrize("closed", [SATURDAY, HOLIDAY], ids=["weekend", "holiday"])
def test_a_closed_day_runs_no_ticker_and_exits_zero(closed, invoked):
    assert rp.main(now_et=closed) == 0
    assert invoked == []


def test_a_closed_day_still_refreshes_run_status(_env, invoked):
    rp.main(now_et=SATURDAY)
    status = load_run_status(_artifacts(_env) / RUN_STATUS_FILENAME)
    assert status is not None
    assert (status.exit_code, status.per_ticker, status.fatal) == (0, {}, False)


def test_a_closed_day_carries_the_last_trading_day_evidence_forward(_env, monkeypatch):
    def failing(ticker: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(graph_mod, "invoke_pipeline", failing)
    friday = datetime(2026, 7, 10, 1, 30, tzinfo=_ET)
    assert rp.main(now_et=friday) == 2
    path = _artifacts(_env) / RUN_STATUS_FILENAME
    before = load_run_status(path)
    assert rp.main(now_et=SATURDAY) == 0
    after = load_run_status(path)
    assert before.exit_code == 2
    assert (after.exit_code, after.run_id) == (before.exit_code, before.run_id)
    assert after.per_ticker_exception == before.per_ticker_exception != {}
    assert after.generated_at_utc >= before.generated_at_utc


def test_a_closed_day_releases_the_lock(invoked):
    assert rp.main(now_et=SATURDAY) == 0
    assert rp.main(now_et=SATURDAY) == 0


def test_a_closed_day_leaves_the_heartbeat_finished(_env, invoked):
    rp.main(now_et=SATURDAY)
    beat = json.loads((_artifacts(_env) / HEARTBEAT_FILENAME).read_text(encoding="utf-8"))
    assert (beat["state"], beat["exit_code"], beat["done"]) == ("finished", 0, 0)


def test_a_trading_day_still_runs_every_ticker(invoked):
    assert rp.main(now_et=TRADING_DAY) == 0
    assert invoked == ["AAPL"]


def test_the_override_runs_a_closed_day(monkeypatch, invoked):
    monkeypatch.setenv("PIPELINE_RUN_NON_TRADING_DAYS", "1")
    assert rp.main(now_et=SATURDAY) == 0
    assert invoked == ["AAPL"]
