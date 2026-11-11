"""Tests for SQLite cache: TTL expiry, CRUD, prefix invalidation, singleton."""

import threading
import time
from pathlib import Path

import pytest

# Project root resolved from this file — never hardcode a machine-specific absolute path
# (a hardcoded Windows cwd broke the CLI subprocess tests on Linux CI).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

from aer.data.cache import _MISSING, TTL_ANALYSIS, TTL_BRIEF, TTL_PRICES, TTL_QUOTE, Cache
from aer.utils.cache import (
    RecommendationLog,
    _DomainCache,
    agent_cache,
    brief_cache,
    data_cache,
    get_recommendation_log_store,
)


class TestConstants:
    def test_ttl_prices_24h(self) -> None:
        assert TTL_PRICES == 24 * 3600

    def test_ttl_quote_1h(self) -> None:
        assert TTL_QUOTE == 1 * 3600

    def test_ttl_analysis_12h(self) -> None:
        assert TTL_ANALYSIS == 12 * 3600

    def test_ttl_brief_equals_analysis(self) -> None:
        assert TTL_BRIEF == TTL_ANALYSIS


class TestGetSet:
    def test_missing_key_returns_none(self, tmp_cache: _DomainCache) -> None:
        assert tmp_cache.get("nonexistent") is None

    def test_round_trip_dict(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("k", {"a": 1, "b": [2, 3]})
        assert tmp_cache.get("k") == {"a": 1, "b": [2, 3]}

    def test_round_trip_list(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("k", [1, 2, 3])
        assert tmp_cache.get("k") == [1, 2, 3]

    def test_round_trip_scalar(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("i", 42)
        assert tmp_cache.get("i") == 42
        tmp_cache.set("f", 3.14)
        assert tmp_cache.get("f") == pytest.approx(3.14)
        tmp_cache.set("s", "hello")
        assert tmp_cache.get("s") == "hello"

    def test_overwrite_existing_key(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("k", "original")
        tmp_cache.set("k", "updated")
        assert tmp_cache.get("k") == "updated"

    def test_expired_entry_returns_none(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("k", "v", ttl=1)
        time.sleep(1.05)
        assert tmp_cache.get("k") is None

    def test_non_expired_entry_returned(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("k", "v", ttl=3600)
        assert tmp_cache.get("k") == "v"

    def test_default_ttl_is_prices(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("k", "v")
        # default TTL is 24h — should not expire in test duration
        assert tmp_cache.get("k") == "v"

    def test_non_json_serialisable_coerced_to_str(self, tmp_cache: _DomainCache) -> None:
        from datetime import date

        d = date(2024, 1, 1)
        tmp_cache.set("k", {"date": d})
        result = tmp_cache.get("k")
        assert result is not None
        assert result["date"] == "2024-01-01"


class TestDelete:
    def test_delete_removes_entry(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("k", "v")
        tmp_cache.delete("k")
        assert tmp_cache.get("k") is None

    def test_delete_nonexistent_no_error(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.delete("no_such_key")

    def test_set_after_delete(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("k", "a")
        tmp_cache.delete("k")
        tmp_cache.set("k", "b")
        assert tmp_cache.get("k") == "b"


class TestContains:
    def test_existing_key(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("k", "v")
        assert "k" in tmp_cache

    def test_missing_key(self, tmp_cache: _DomainCache) -> None:
        assert "missing" not in tmp_cache

    def test_expired_key_not_in_cache(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("k", "v", ttl=1)
        time.sleep(1.05)
        assert "k" not in tmp_cache


class TestCleanup:
    def test_removes_expired_entries(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("e1", "a", ttl=1)
        tmp_cache.set("e2", "b", ttl=1)
        tmp_cache.set("keep", "c", ttl=3600)
        time.sleep(1.05)
        deleted = tmp_cache.cleanup()
        assert deleted == 2
        assert tmp_cache.get("keep") == "c"

    def test_returns_count(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("x", "v", ttl=1)
        time.sleep(1.05)
        assert tmp_cache.cleanup() >= 1

    def test_empty_cache_returns_zero(self, tmp_cache: _DomainCache) -> None:
        assert tmp_cache.cleanup() == 0

    def test_no_expired_entries_returns_zero(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("k", "v", ttl=3600)
        assert tmp_cache.cleanup() == 0


class TestInvalidatePrefix:
    def test_removes_matching_keys(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("agent:fundamental:AAPL", "v")
        tmp_cache.set("agent:fundamental:MSFT", "v")
        tmp_cache.set("prices:AAPL", "v")
        deleted = tmp_cache.invalidate_prefix("agent:fundamental:")
        assert deleted == 2
        assert tmp_cache.get("prices:AAPL") == "v"

    def test_returns_count(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("pfx:a", "v")
        tmp_cache.set("pfx:b", "v")
        assert tmp_cache.invalidate_prefix("pfx:") == 2

    def test_no_match_returns_zero(self, tmp_cache: _DomainCache) -> None:
        tmp_cache.set("other:key", "v")
        assert tmp_cache.invalidate_prefix("nomatch:") == 0


class TestThreadSafety:
    def test_concurrent_writes_no_corruption(self, tmp_cache: _DomainCache) -> None:
        errors: list[Exception] = []

        def writer(n: int) -> None:
            try:
                for i in range(10):
                    tmp_cache.set(f"thread{n}:key{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestPelosiCacheRoundtrip:
    def test_pelosi_cache_roundtrip(self, tmp_cache: _DomainCache) -> None:
        """pelosi:trades:{n} key round-trips through SQLite cache."""
        trades = [{"ticker": "NVDA", "amount": "1001-15000", "filed_date": "2026-04-10"}]
        tmp_cache.set("pelosi:trades:90", trades, ttl=21600)
        result = tmp_cache.get("pelosi:trades:90")
        assert result == trades


class TestCreatedAtFor:
    def test_missing_key_returns_none(self, tmp_cache: _DomainCache) -> None:
        assert tmp_cache.created_at_for("nonexistent") is None

    def test_existing_key_returns_float(self, tmp_cache: _DomainCache) -> None:
        import time

        before = time.time()
        tmp_cache.set("k", {"x": 1}, ttl=3600)
        result = tmp_cache.created_at_for("k")
        assert isinstance(result, float)
        assert before <= result <= time.time() + 1

    def test_expired_key_returns_none(self, tmp_cache: _DomainCache) -> None:
        import time

        tmp_cache.set("exp", {"x": 1}, ttl=1)
        time.sleep(1.05)
        assert tmp_cache.created_at_for("exp") is None


class TestGetCacheSingleton:
    def test_same_instance_returned(self, tmp_path) -> None:
        from unittest.mock import MagicMock, patch

        import aer.data.cache as cache_mod

        original = cache_mod._default_cache
        cache_mod._default_cache = None
        mock_settings = MagicMock()
        mock_settings.cache_db_path = str(tmp_path / "singleton.db")
        try:
            with patch("aer.config.get_settings", return_value=mock_settings):
                a = cache_mod.get_cache()
                b = cache_mod.get_cache()
            assert a is b
        finally:
            cache_mod._default_cache = original


class TestEnsureSchema:
    def test_ensure_schema_idempotent(self, tmp_path) -> None:
        """Calling _ensure_schema twice on the same DB must not raise."""
        c = Cache(db_path=str(tmp_path / "idempotent.db"))
        c._ensure_schema()  # second call — must not fail
        assert c.get("nonexistent") is _MISSING

    def test_schema_version_table_created(self, tmp_path) -> None:
        c = Cache(db_path=str(tmp_path / "schema.db"))
        conn = c._connect()
        row = conn.execute("SELECT version FROM schema_version WHERE version = 1").fetchone()
        assert row is not None and row[0] == 1

    def test_domain_column_exists(self, tmp_path) -> None:
        c = Cache(db_path=str(tmp_path / "domain.db"))
        conn = c._connect()
        info = conn.execute("PRAGMA table_info(cache)").fetchall()
        col_names = [col[1] for col in info]
        assert "domain" in col_names


class TestSetWithDomain:
    def test_set_with_domain_writes_domain_column(self, tmp_data_cache: _DomainCache) -> None:
        tmp_data_cache.set("dk1", "value1", ttl=3600)
        conn = tmp_data_cache._cache._connect()
        row = conn.execute("SELECT domain FROM cache WHERE key = ?", ("dk1",)).fetchone()
        assert row is not None
        assert row[0] == "data"

    def test_set_with_domain_agent(self, tmp_agent_cache: _DomainCache) -> None:
        tmp_agent_cache.set("ak1", {"x": 1}, ttl=43200)
        conn = tmp_agent_cache._cache._connect()
        row = conn.execute("SELECT domain FROM cache WHERE key = ?", ("ak1",)).fetchone()
        assert row is not None
        assert row[0] == "agent"

    def test_set_with_domain_upsert_updates_domain(self, tmp_path) -> None:
        underlying = Cache(db_path=str(tmp_path / "upsert.db"))
        dc_data = _DomainCache(underlying, TTL_PRICES, "data")
        dc_agent = _DomainCache(underlying, TTL_ANALYSIS, "agent")
        dc_data.set("key1", "v1", ttl=3600)
        dc_agent.set("key1", "v2", ttl=3600)
        conn = underlying._connect()
        row = conn.execute("SELECT value, domain FROM cache WHERE key = ?", ("key1",)).fetchone()
        assert row is not None
        assert row[1] == "agent"

    def test_direct_cache_set_raises(self, tmp_path) -> None:
        """Cache.set() must raise RuntimeError to enforce domain tagging via _DomainCache."""
        raw = Cache(db_path=str(tmp_path / "guard.db"))
        with pytest.raises(RuntimeError, match="domain tagging"):
            raw.set("legacykey", "val", ttl=3600)


class TestDomainCacheClass:
    def test_set_writes_correct_domain(self, tmp_path) -> None:
        backing = Cache(db_path=str(tmp_path / "dc.db"))
        dc = _DomainCache(backing, TTL_PRICES, "data")
        dc.set("k1", "hello")
        conn = backing._connect()
        row = conn.execute("SELECT domain FROM cache WHERE key = ?", ("k1",)).fetchone()
        assert row is not None and row[0] == "data"

    def test_get_returns_stored_value(self, tmp_path) -> None:
        backing = Cache(db_path=str(tmp_path / "dc.db"))
        dc = _DomainCache(backing, TTL_PRICES, "data")
        dc.set("k2", {"a": 1})
        assert dc.get("k2") == {"a": 1}

    def test_delete_removes_key(self, tmp_path) -> None:
        backing = Cache(db_path=str(tmp_path / "dc.db"))
        dc = _DomainCache(backing, TTL_PRICES, "data")
        dc.set("k3", "v")
        dc.delete("k3")
        assert dc.get("k3") is None

    def test_set_with_no_ttl_uses_default_ttl_int(self, tmp_path) -> None:
        backing = Cache(db_path=str(tmp_path / "dc.db"))
        dc = _DomainCache(backing, 7200, "data")
        dc.set("k4", "v")
        ttl = backing.ttl_for("k4")
        assert ttl == 7200

    def test_set_with_callable_default_ttl_resolves_at_call_time(self, tmp_path) -> None:
        backing = Cache(db_path=str(tmp_path / "dc.db"))
        call_count = [0]

        def dynamic_ttl() -> int:
            call_count[0] += 1
            return 9999

        dc = _DomainCache(backing, dynamic_ttl, "data")
        dc.set("k5", "v")
        assert call_count[0] == 1
        assert backing.ttl_for("k5") == 9999

    def test_set_with_explicit_ttl_overrides_default(self, tmp_path) -> None:
        backing = Cache(db_path=str(tmp_path / "dc.db"))
        dc = _DomainCache(backing, TTL_PRICES, "data")
        dc.set("k6", "v", ttl=1234)
        assert backing.ttl_for("k6") == 1234

    def test_flush_returns_correct_rowcount(self, tmp_path) -> None:
        backing = Cache(db_path=str(tmp_path / "dc.db"))
        dc_data = _DomainCache(backing, TTL_PRICES, "data")
        dc_agent = _DomainCache(backing, TTL_ANALYSIS, "agent")
        dc_data.set("d1", "v1")
        dc_data.set("d2", "v2")
        dc_agent.set("a1", "v3")
        deleted = dc_data.flush()
        assert deleted == 2

    def test_flush_only_deletes_own_domain(self, tmp_path) -> None:
        backing = Cache(db_path=str(tmp_path / "dc.db"))
        dc_data = _DomainCache(backing, TTL_PRICES, "data")
        dc_agent = _DomainCache(backing, TTL_ANALYSIS, "agent")
        dc_data.set("d1", "v1")
        dc_agent.set("a1", "v2")
        dc_data.flush()
        assert dc_agent.get("a1") == "v2"
        assert dc_data.get("d1") is None

    def test_flush_empty_domain_returns_zero(self, tmp_path) -> None:
        backing = Cache(db_path=str(tmp_path / "dc.db"))
        dc = _DomainCache(backing, TTL_PRICES, "data")
        assert dc.flush() == 0


class TestDomainCacheSingletons:
    def test_data_cache_singleton_identity(self, tmp_path) -> None:
        import aer.utils.cache as utils_mod

        original = utils_mod._data_cache
        utils_mod._data_cache = None
        try:
            a = data_cache()
            b = data_cache()
            assert a is b
        finally:
            utils_mod._data_cache = original

    def test_agent_cache_singleton_identity(self, tmp_path) -> None:
        import aer.utils.cache as utils_mod

        original = utils_mod._agent_cache
        utils_mod._agent_cache = None
        try:
            a = agent_cache()
            b = agent_cache()
            assert a is b
        finally:
            utils_mod._agent_cache = original

    def test_brief_cache_singleton_identity(self, tmp_path) -> None:
        import aer.utils.cache as utils_mod

        original = utils_mod._brief_cache
        utils_mod._brief_cache = None
        try:
            a = brief_cache()
            b = brief_cache()
            assert a is b
        finally:
            utils_mod._brief_cache = original

    def test_data_cache_domain_is_data(self) -> None:
        dc = data_cache()
        assert dc._domain == "data"

    def test_agent_cache_domain_is_agent(self) -> None:
        ac = agent_cache()
        assert ac._domain == "agent"

    def test_brief_cache_domain_is_brief(self) -> None:
        bc = brief_cache()
        assert bc._domain == "brief"

    def test_brief_cache_ttl_is_callable(self) -> None:
        bc = brief_cache()
        assert callable(bc._default_ttl)


class TestRecommendationLog:
    def test_log_and_get_log_round_trip(self, tmp_path) -> None:
        backing = Cache(db_path=str(tmp_path / "rec.db"))
        rl = RecommendationLog(backing)
        rl.log("AAPL", "BUY", 0.8, 195.5)
        records = rl.get_log(days=7)
        assert len(records) == 1
        assert records[0]["ticker"] == "AAPL"
        assert records[0]["recommendation"] == "BUY"
        assert records[0]["conviction"] == pytest.approx(0.8)

    def test_get_log_empty_returns_empty_list(self, tmp_path) -> None:
        backing = Cache(db_path=str(tmp_path / "rec.db"))
        rl = RecommendationLog(backing)
        assert rl.get_log() == []

    def test_get_recommendation_log_store_singleton(self) -> None:
        import aer.utils.cache as utils_mod

        original = utils_mod._rec_log
        utils_mod._rec_log = None
        try:
            a = get_recommendation_log_store()
            b = get_recommendation_log_store()
            assert a is b
        finally:
            utils_mod._rec_log = original


class TestGetOrComputeWithDomainCache:
    def test_get_or_compute_explicit_cache_used(self, tmp_path) -> None:
        from aer.utils.cache import get_or_compute

        backing = Cache(db_path=str(tmp_path / "oc.db"))
        dc = _DomainCache(backing, TTL_PRICES, "agent")
        called = [0]

        def compute() -> str:
            called[0] += 1
            return "computed"

        result = get_or_compute("oc_key", compute, ttl=3600, cache=dc)
        assert result == "computed"
        assert called[0] == 1
        # Second call should use cache, not recompute
        result2 = get_or_compute("oc_key", compute, ttl=3600, cache=dc)
        assert result2 == "computed"
        assert called[0] == 1

    def test_get_or_compute_domain_column_written(self, tmp_path) -> None:
        from aer.utils.cache import get_or_compute

        backing = Cache(db_path=str(tmp_path / "oc2.db"))
        dc = _DomainCache(backing, TTL_PRICES, "agent")
        get_or_compute("domkey", lambda: 42, ttl=3600, cache=dc)
        conn = backing._connect()
        row = conn.execute("SELECT domain FROM cache WHERE key = ?", ("domkey",)).fetchone()
        assert row is not None and row[0] == "agent"


class TestFlushCLI:
    def test_flush_cli_data_domain(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "aer.utils.cache", "flush", "data"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "Flushed" in result.stdout
        assert "data cache" in result.stdout

    def test_flush_cli_all_domains(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "aer.utils.cache", "flush", "all"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert result.stdout.count("Flushed") >= 4  # one per domain

    def test_flush_cli_invalid_domain(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "aer.utils.cache", "flush", "unknown"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 1

    def test_flush_cli_no_domain(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "aer.utils.cache", "flush"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
        )
        assert result.returncode == 1
