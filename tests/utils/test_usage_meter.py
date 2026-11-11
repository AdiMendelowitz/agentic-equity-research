"""Token-aware metering: provider-reported usage per call, one summary line per run, zero extra calls."""

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aer.utils import llm_client, llm_providers, usage_meter
from aer.utils.constants import (
    GROQ_FREE_REQUESTS_PER_DAY,
    GROQ_FREE_TOKENS_PER_DAY,
    USAGE_BUDGET_WARN_FRACTION,
)


@pytest.fixture(autouse=True)
def _fresh_meter():
    usage_meter.reset()
    yield
    usage_meter.reset()


def _openai_response(content, prompt_tokens=None, completion_tokens=None):
    usage = (
        None
        if prompt_tokens is None
        else SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


def test_budget_constants_match_the_documented_groq_free_plan():
    assert GROQ_FREE_TOKENS_PER_DAY == 200_000
    assert GROQ_FREE_REQUESTS_PER_DAY == 1_000
    assert USAGE_BUDGET_WARN_FRACTION == 0.8


def test_record_aggregates_per_provider_and_model():
    usage_meter.record("groq", "m1", 100, 20)
    usage_meter.record("groq", "m1", 50, 10)
    usage_meter.record("anthropic", "haiku", 7, 3, cache_read=5, cache_write=2)
    rows = {(r["provider"], r["model"]): r for r in usage_meter.snapshot()}
    assert rows[("groq", "m1")] == {
        "provider": "groq",
        "model": "m1",
        "calls": 2,
        "in_tokens": 150,
        "out_tokens": 30,
        "cache_read": 0,
        "cache_write": 0,
        "calls_without_usage": 0,
    }
    assert rows[("anthropic", "haiku")]["cache_read"] == 5


def test_a_call_without_usage_is_counted_not_guessed():
    usage_meter.record_openai_usage("nvidia_nim", "m", _openai_response("x"))
    (row,) = usage_meter.snapshot()
    assert (row["calls"], row["in_tokens"], row["calls_without_usage"]) == (1, 0, 1)


def test_last_usage_is_per_thread():
    usage_meter.record("groq", "m", 11, 2)
    seen = {}
    t = threading.Thread(target=lambda: seen.setdefault("other", usage_meter.last_usage()))
    t.start()
    t.join()
    assert usage_meter.last_usage() == (11, 2)
    assert seen["other"] == (None, None)


def test_budget_warning_fires_once_at_the_threshold(caplog):
    caplog.set_level(logging.WARNING, logger="aer.utils.usage_meter")
    below = int(GROQ_FREE_TOKENS_PER_DAY * USAGE_BUDGET_WARN_FRACTION) - 1
    usage_meter.record("groq", "m", below, 0)
    assert not caplog.records
    usage_meter.record("groq", "m", 1, 0)
    usage_meter.record("groq", "m", 1, 0)
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1 and "groq" in msgs[0] and "200000" in msgs[0]


def test_write_run_summary_appends_one_line_with_totals(tmp_path):
    usage_meter.record("groq", "m", 10, 5)
    path = tmp_path / "llm_usage.jsonl"
    usage_meter.write_run_summary(path, run_id="r1")
    usage_meter.write_run_summary(path, run_id="r2")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [x["run_id"] for x in lines] == ["r1", "r2"]
    assert lines[0]["totals"] == {"calls": 1, "in_tokens": 10, "out_tokens": 5}
    assert lines[0]["by_model"][0]["model"] == "m"


def test_every_free_provider_records_its_usage_even_for_an_empty_answer(monkeypatch):
    for getter, fn, name in (
        ("_get_openrouter_client", "call_openrouter", "openrouter"),
        ("_get_nim_client", "call_nvidia_nim", "nvidia_nim"),
        ("_get_groq_client", "call_groq", "groq"),
    ):
        client = MagicMock()
        client.chat.completions.create.return_value = _openai_response("", 40, 0)
        monkeypatch.setattr(llm_providers, getter, lambda c=client: c)
        kwargs = {"json_mode": False} if name != "nvidia_nim" else {}
        with pytest.raises(ValueError):
            getattr(llm_providers, fn)("m", "s", "u", max_tokens=5, timeout=1.0, **kwargs)
    rows = {r["provider"]: r for r in usage_meter.snapshot()}
    assert {p: (r["calls"], r["in_tokens"]) for p, r in rows.items()} == {
        "openrouter": (1, 40),
        "nvidia_nim": (1, 40),
        "groq": (1, 40),
    }


def test_complete_returns_the_free_provider_token_counts(monkeypatch):
    def fake_attempt(**kwargs):
        usage_meter.record(kwargs["provider"], kwargs["model"], 321, 45)
        return "text"

    monkeypatch.setattr(llm_client, "_attempt_provider", fake_attempt)
    r = llm_client.complete("s", "u", tier="ANALYSIS", max_tokens=100)
    assert (r.input_tokens, r.output_tokens) == (321, 45)


def test_a_billed_anthropic_call_with_no_text_is_still_metered(monkeypatch):
    response = SimpleNamespace(content=[], usage=SimpleNamespace(input_tokens=90, output_tokens=0))
    client = MagicMock()
    client.messages.create.return_value = response
    monkeypatch.setattr(llm_client, "_get_anthropic_client", lambda: client)
    with pytest.raises(ValueError):
        llm_client._call_anthropic_fallback(
            model="claude-haiku-4-5-20251001", system_prompt="s", user_message="u", max_tokens=10
        )
    (row,) = usage_meter.snapshot()
    assert (row["provider"], row["in_tokens"]) == ("anthropic", 90)


def test_a_non_integer_usage_field_is_treated_as_missing():
    usage_meter.record_openai_usage(
        "groq", "m", SimpleNamespace(usage=SimpleNamespace(prompt_tokens="12", completion_tokens=None))
    )
    (row,) = usage_meter.snapshot()
    assert (row["in_tokens"], row["calls_without_usage"]) == (0, 1)
