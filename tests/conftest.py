"""Shared fixtures for the agentic-equity-research test suite."""

from __future__ import annotations

import socket
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

import aer.utils.cache as _aer_utils_cache
from aer.data.cache import Cache
from aer.utils.cache import TTL_ANALYSIS, TTL_BRIEF, TTL_PRICES, _DomainCache, market_aware_ttl

# identity of the real factories, captured once at collection, so real_key_cache can find
# every consumer module that bound them at import time.
_ORIGINAL_FACTORIES = {
    _n: getattr(_aer_utils_cache, _n) for _n in ("data_cache", "agent_cache", "brief_cache", "news_cache", "lseg_cache")
}


@pytest.fixture
def tmp_cache(tmp_path) -> _DomainCache:
    """Fresh _DomainCache backed by a temp SQLite file. Domain 'test' for isolation."""
    return _DomainCache(Cache(db_path=str(tmp_path / "test.db")), TTL_PRICES, "test")


@pytest.fixture
def tmp_data_cache(tmp_path) -> _DomainCache:
    """Fresh data-domain _DomainCache backed by a temp SQLite file."""
    from aer.data.cache import Cache

    return _DomainCache(Cache(db_path=str(tmp_path / "test.db")), TTL_PRICES, "data")


@pytest.fixture
def tmp_agent_cache(tmp_path) -> _DomainCache:
    """Fresh agent-domain _DomainCache backed by a temp SQLite file."""
    from aer.data.cache import Cache

    return _DomainCache(Cache(db_path=str(tmp_path / "test.db")), TTL_ANALYSIS, "agent")


@pytest.fixture
def tmp_brief_cache(tmp_path) -> _DomainCache:
    """Fresh brief-domain _DomainCache backed by a temp SQLite file. Callable TTL."""
    from aer.data.cache import Cache

    return _DomainCache(
        Cache(db_path=str(tmp_path / "test.db")),
        lambda: market_aware_ttl(TTL_BRIEF),
        "brief",
    )


@pytest.fixture(autouse=True)
def mock_aer_log_dir(monkeypatch, tmp_path):
    """Test isolation: redirect AER_LOG_DIR to tmp_path. Autouse to isolate all log writes."""

    aer_log_dir = tmp_path / "logs"
    aer_log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AER_LOG_DIR", str(aer_log_dir))
    yield aer_log_dir


@pytest.fixture(autouse=True)
def mock_aer_artifact_dir(monkeypatch, tmp_path):
    """Test isolation: redirect AER_ARTIFACT_DIR to tmp_path. Autouse to isolate all artifact writes."""
    aer_artifact_dir = tmp_path / "artifacts"
    aer_artifact_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AER_ARTIFACT_DIR", str(aer_artifact_dir))
    yield aer_artifact_dir


@pytest.fixture(autouse=True)
def mock_cache_db_path(monkeypatch, tmp_path):
    """Test isolation: cache DB to tmp_path: no test touches .cache/aer.db."""
    from aer.config import get_settings, reset_settings_cache
    from aer.data.cache import reset_cache

    db_path = tmp_path / "test_cache.db"
    monkeypatch.setenv("CACHE_DB_PATH", str(db_path))
    reset_settings_cache()
    reset_cache()
    # runtime proof
    assert str(tmp_path) in get_settings().cache_db_path
    yield db_path
    reset_settings_cache()
    reset_cache()


# ---------------------------------------------------------------------------
# real_key_cache fixture
#
# Monkeypatches all 5 domain-cache factories in aer.utils.cache to tmp-SQLite
# instances AND patches the external network boundaries (yfinance, fredapi,
# requests) with canned minimal payloads so each get_* fetcher executes its
# real body (including the real cache.set(key, ...) call) without network I/O.
#
# Exposes written_keys() -> set[str] so contract tests can assert that a
# SourceSpec.cache_key_template.format(...) matches an actually-written key.
# This is the load-bearing defence: tests exercise real fetcher keying,
# not only conftest fakes.
# ---------------------------------------------------------------------------


class _RealKeyCache:
    """Wrapper around a shared tmp SQLite that exposes written_keys()."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def written_keys(self) -> set[str]:
        """Return all cache keys that were written during the fixture's lifetime."""
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute("SELECT key FROM cache").fetchall()
            return {row[0] for row in rows}
        finally:
            conn.close()


def _make_domain_cache(db_path: str, ttl: Any, domain: str) -> _DomainCache:
    return _DomainCache(Cache(db_path=db_path), ttl, domain)


# Canned yfinance payloads: minimal but shaped so fetcher bodies reach cache.set.
_CANNED_OHLCV_DICT = {
    "columns": ["Open", "High", "Low", "Close", "Volume"],
    "index": ["2026-01-02"],
    "data": [[150.0, 152.0, 149.0, 151.0, 1000000]],
}
_CANNED_INFO = {
    "shortName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "marketCap": 3_000_000_000_000,
    "trailingPE": 28.5,
    "forwardPE": 25.0,
    "priceToBook": 40.0,
    "returnOnEquity": 1.50,
    "debtToEquity": 120.0,
    "currentRatio": 0.94,
    "quickRatio": 0.87,
    "ebitdaMargins": 0.33,
    "revenueGrowth": 0.05,
    "grossMargins": 0.44,
    "operatingMargins": 0.30,
    "targetMeanPrice": 210.0,
    "recommendationKey": "buy",
    "heldPercentInstitutions": 0.60,
    "shortPercentOfFloat": 0.007,
    "insiderHoldPercent": 0.003,
    "sharesOutstanding": 15_000_000_000,
}
_CANNED_FAST_INFO = MagicMock()
_CANNED_FAST_INFO.last_price = 151.0

_CANNED_FRED_SERIES = pd.Series(
    [4.33, 4.50, 4.75],
    index=pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
    name="FEDFUNDS",
)

_CANNED_AW_PAGE: list[dict] = [
    {
        "ticker": "AAPL",
        "name": "Apple",
        "rank": 1,
        "rank_24h_ago": 2,
        "mentions": 50,
        "upvotes": 120,
    }
]


@pytest.fixture
def real_key_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _RealKeyCache:
    """Execute real get_* fetcher bodies against a tmp SQLite and expose written keys.

    All 5 domain-cache factories are patched to the same tmp-backed SQLite so
    written_keys() collects keys from every domain in one query.

    Network boundaries (yfinance/fredapi/requests) are patched to canned minimal
    payloads that are shaped to let each fetcher body reach its cache.set() call.

    This fixture is the load-bearing gate for  contract tests: tests compare
    SourceSpec.cache_key_template.format(...) against real keys written here.
    """
    db_path = str(tmp_path / "real_key_test.db")
    ttl_prices = TTL_PRICES
    ttl_analysis = TTL_ANALYSIS

    data_dc = _make_domain_cache(db_path, ttl_prices, "data")
    agent_dc = _make_domain_cache(db_path, ttl_analysis, "agent")
    brief_dc = _make_domain_cache(db_path, ttl_analysis, "brief")
    news_dc = _make_domain_cache(db_path, ttl_prices, "news")
    lseg_dc = _make_domain_cache(db_path, ttl_analysis, "lseg")

    # Patch domain-cache factories in the cache utils module.
    monkeypatch.setattr("aer.utils.cache.data_cache", lambda: data_dc)
    monkeypatch.setattr("aer.utils.cache.agent_cache", lambda: agent_dc)
    monkeypatch.setattr("aer.utils.cache.brief_cache", lambda: brief_dc)
    monkeypatch.setattr("aer.utils.cache.news_cache", lambda: news_dc)
    monkeypatch.setattr("aer.utils.cache.lseg_cache", lambda: lseg_dc)

    # Patch factories as imported in individual data modules (each module cached the import).
    for mod in (
        "aer.data.prices",
        "aer.data.financials",
        "aer.data.earnings",
        "aer.data.ownership",
    ):
        monkeypatch.setattr(f"{mod}.data_cache", lambda: data_dc, raising=False)

    for mod in (
        "aer.data.options_flow",
        "aer.data.sentiment_feeds",
    ):
        monkeypatch.setattr(f"{mod}.agent_cache", lambda: agent_dc, raising=False)

    monkeypatch.setattr("aer.data.macro.data_cache", lambda: data_dc, raising=False)
    monkeypatch.setattr("aer.data.breadth.data_cache", lambda: data_dc, raising=False)
    monkeypatch.setattr("aer.data.regime.data_cache", lambda: data_dc, raising=False)
    monkeypatch.setattr("aer.data.news_feeds.news_cache", lambda: news_dc, raising=False)

    # the hand list above reaches only some of the ~40 modules that bind a factory at import
    # (`from aer.utils.cache import data_cache`). Sweep every loaded aer.* module and rebind any
    # attribute that still IS an original factory; modules imported after this point bind the
    # patched attribute of aer.utils.cache instead, so no consumer can reach the real cache.
    import sys as _sys

    import aer.utils.cache as _uc

    _replacements = {
        id(_ORIGINAL_FACTORIES["data_cache"]): lambda: data_dc,
        id(_ORIGINAL_FACTORIES["agent_cache"]): lambda: agent_dc,
        id(_ORIGINAL_FACTORIES["brief_cache"]): lambda: brief_dc,
        id(_ORIGINAL_FACTORIES["news_cache"]): lambda: news_dc,
        id(_ORIGINAL_FACTORIES["lseg_cache"]): lambda: lseg_dc,
    }
    for _name, _mod in list(_sys.modules.items()):
        if not (_name == "aer" or _name.startswith("aer.")) or _mod is None or _mod is _uc:
            continue
        for _attr in ("data_cache", "agent_cache", "brief_cache", "news_cache", "lseg_cache"):
            _val = getattr(_mod, _attr, None)
            if _val is not None and id(_val) in _replacements:
                monkeypatch.setattr(_mod, _attr, _replacements[id(_val)])

    # Build a minimal fake yfinance.Ticker that returns canned data.
    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self._symbol = symbol

        @property
        def info(self) -> dict:
            return dict(_CANNED_INFO)

        @property
        def fast_info(self) -> Any:
            return _CANNED_FAST_INFO

        @property
        def calendar(self) -> dict:
            return {}

        @property
        def earnings_dates(self) -> pd.DataFrame:
            return pd.DataFrame()

        @property
        def options(self) -> tuple:
            return ()

        @property
        def institutional_holders(self) -> pd.DataFrame:
            return pd.DataFrame()

        @property
        def major_holders(self) -> pd.DataFrame:
            return pd.DataFrame()

        @property
        def insider_transactions(self) -> pd.DataFrame:
            return pd.DataFrame()

        @property
        def recommendations(self) -> pd.DataFrame:
            return pd.DataFrame()

    def _fake_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        df = pd.DataFrame(
            _CANNED_OHLCV_DICT["data"],
            columns=_CANNED_OHLCV_DICT["columns"],
            index=pd.to_datetime(_CANNED_OHLCV_DICT["index"]),
        )
        df.index.name = "Date"
        return df

    monkeypatch.setattr("aer.data.prices.yf.Ticker", _FakeTicker)
    monkeypatch.setattr("aer.data.prices.yf.download", _fake_download)
    monkeypatch.setattr("aer.data.earnings.yf.Ticker", _FakeTicker, raising=False)
    monkeypatch.setattr("aer.data.ownership.yf.Ticker", _FakeTicker, raising=False)
    monkeypatch.setattr("aer.data.options_flow.yf.Ticker", _FakeTicker, raising=False)
    monkeypatch.setattr("aer.data.breadth.yf.download", _fake_download, raising=False)

    # Patch fredapi.Fred so macro.get_series() writes its key.
    class _FakeFred:
        def __init__(self, api_key: str = "") -> None:
            pass

        def get_series(self, series_id: str, **kwargs: Any) -> pd.Series:
            return _CANNED_FRED_SERIES.copy().rename(series_id)

    monkeypatch.setattr("aer.data.macro.Fred", _FakeFred)

    # Patch requests.get for ApeWisdom so sentiment fetch writes its key.
    def _fake_requests_get(url: str, **kwargs: Any) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "apewisdom" in url:
            resp.json.return_value = {"results": _CANNED_AW_PAGE}
        else:
            resp.json.return_value = {}
            resp.text = ""
            resp.content = b""
            resp.status_code = 200
        return resp

    monkeypatch.setattr("aer.data.sentiment_feeds.requests.get", _fake_requests_get, raising=False)
    monkeypatch.setattr("aer.data.news_feeds.requests.get", _fake_requests_get, raising=False)

    return _RealKeyCache(db_path)


# ---------------------------------------------------------------------------
# fake_fetcher_counter fixture
#
# A callable factory returning a fetcher stub that:
#   - increments a Counter per call (tracks call counts for idempotency tests)
#   - writes a caller-supplied key on first call only (cache-first simulation)
#
# Used by driver idempotency tests (second-pass zero-calls assertion).
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_fetcher_counter() -> Any:
    """Return a factory that produces per-key call-counting fake fetchers.

    Usage:
        factory = fake_fetcher_counter
        fetcher, calls = factory("ownership:AAPL", cache)
        # call fetcher("AAPL") twice: cache hit on second call → calls["AAPL"] == 1
    """

    def _make_fetcher(key: str, cache: _DomainCache) -> tuple[Any, Counter]:
        call_counts: Counter = Counter()

        def _fetcher(*args: Any, **kwargs: Any) -> dict:
            ticker_key = args[0] if args else "_global"
            call_counts[ticker_key] += 1
            # Write key on first call only (cache-first simulation).
            if call_counts[ticker_key] == 1:
                cache.set(key, {"stub": True})
            return {"stub": True}

        return _fetcher, call_counts

    return _make_fetcher


@pytest.fixture(autouse=True)
def _opinion_ticker_guard_off(monkeypatch):
    """the production writer admits only watchlist and held tickers; suite fixtures use synthetic
    tickers against a tmp database. tests/test_forward_return_admission.py switches the guard back on."""
    monkeypatch.setattr("aer.data.opinions.OPINION_TICKER_GUARD_ENABLED", False)


@pytest.fixture(autouse=True)
def _isolate_ml_model_dir(tmp_path, monkeypatch):
    """aer.ml.models saved AAPL_<date>.joblib into the production ml_models/ on every gate day and pruned
    that ticker's files older than 7 days. Both modules bind MODEL_DIR at import, so both are redirected.
    Tests that patch MODEL_DIR themselves still win (their patch is applied after this one)."""
    import aer.ml.constants as mlc
    import aer.ml.models as mlm

    d = tmp_path / "ml_models"
    monkeypatch.setattr(mlc, "MODEL_DIR", d)
    monkeypatch.setattr(mlm, "MODEL_DIR", d)


@pytest.fixture(autouse=True)
def _reset_ollama_health_cache():
    """is_ollama_available() remembers a failure for 300 s; each test starts with no memory of another's."""
    from aer.utils.ollama_client import reset_ollama_health_cache

    reset_ollama_health_cache()
    yield
    reset_ollama_health_cache()


@pytest.fixture(autouse=True)
def _isolate_unclassified_conflicts_log(tmp_path, monkeypatch):
    """conflict_classifier appends to logs/unclassified_conflicts.jsonl; each test writes to a temporary file."""
    import aer.utils.conflict_classifier as cc

    monkeypatch.setattr(cc, "_UNCLASSIFIED_LOG", tmp_path / "unclassified_conflicts.jsonl")


@pytest.fixture(autouse=True)
def _isolate_prefetch_logs(tmp_path, monkeypatch):
    """every prefetch runtime-state file goes under tmp_path.

    completeness.jsonl is production state; tests write it under tmp_path instead.
    """
    import aer.prefetch.completeness as comp

    d = tmp_path / "prefetch_logs"
    monkeypatch.setattr(comp, "_PREFETCH_LOGS_DIR", d)
    monkeypatch.setattr(comp, "_COMPLETENESS_JSONL", d / "completeness.jsonl")


# ---------------------------------------------------------------------------
# Socket guard
#
# A test reaches no Ollama daemon (port 11434, any host) unless marked ``live``, and no
# non-loopback host unless marked ``integration`` or ``live``. Libraries that open sockets
# outside Python (curl_cffi, used by yfinance) honour the proxy variables, which point at a
# loopback port the guard itself refuses for guarded tests. A blocked attempt raises ConnectionRefusedError, the
# same error a stopped service gives, so code under test takes its normal "unavailable" path.
# ---------------------------------------------------------------------------
_OLLAMA_PORT = 11434
_PROXY_SINK_PORT = 9
_PROXY_SINK = f"http://127.0.0.1:{_PROXY_SINK_PORT}"
_BLOCKED: list[tuple[str, str]] = []
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex


class BlockedConnection(ConnectionRefusedError):
    """Raised by the socket guard; subclass of ConnectionRefusedError on purpose."""


def _is_loopback(host) -> bool:
    import ipaddress

    if isinstance(host, str) and host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False  # a hostname other than localhost: treated as external


def _guard_decision(addr, *, allow_daemon: bool, allow_external: bool) -> str | None:
    """Return a reason to block, or None to allow. Pure: unit-tested directly."""
    if not isinstance(addr, tuple) or len(addr) < 2:
        return None  # AF_UNIX and friends
    host, port = addr[0], addr[1]
    if port == _OLLAMA_PORT and not allow_daemon:
        return f"ollama {host}:{port}"
    if port == _PROXY_SINK_PORT and _is_loopback(host) and not allow_external:
        # refused here, before the OS connect: Windows takes ~2 s to refuse a closed local port
        return f"external via-proxy-sink {host}:{port}"
    if not _is_loopback(host) and not allow_external:
        return f"external {host}:{port}"
    return None


@pytest.fixture(autouse=True)
def _socket_guard(request, monkeypatch):
    live = request.node.get_closest_marker("live") is not None
    integration = request.node.get_closest_marker("integration") is not None
    allow_daemon, allow_external = live, (live or integration)
    if not allow_external:
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(var, _PROXY_SINK)
        monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,::1")
        monkeypatch.setenv("no_proxy", "localhost,127.0.0.1,::1")
    test_id = request.node.nodeid

    def _check(addr):
        why = _guard_decision(addr, allow_daemon=allow_daemon, allow_external=allow_external)
        if why:
            _BLOCKED.append((test_id, why))
            raise BlockedConnection(f"socket guard: {why} blocked; mark the test live or integration")

    def connect(self, addr):
        _check(addr)
        return _REAL_CONNECT(self, addr)

    def connect_ex(self, addr):
        _check(addr)
        return _REAL_CONNECT_EX(self, addr)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    if sys.platform == "win32":  # the Proactor loop connects via ConnectEx, never socket.connect
        import asyncio.windows_events as _we

        _real_proactor_connect = _we.IocpProactor.connect

        def proactor_connect(self, conn, address):
            _check(address)
            return _real_proactor_connect(self, conn, address)

        monkeypatch.setattr(_we.IocpProactor, "connect", proactor_connect)


def pytest_terminal_summary(terminalreporter):
    """Evidence line for the gate: blocked attempts by kind; guarded tests made 0 external connections."""
    kinds = {}
    for _t, why in _BLOCKED:
        k = why.split(" ", 1)[0]
        kinds[k] = kinds.get(k, 0) + 1
    tests = len({t for t, _ in _BLOCKED})
    terminalreporter.write_line(
        f"socket guard: blocked {len(_BLOCKED)} attempts from {tests} tests {kinds} "
        "(guarded paths: socket.connect, connect_ex, Windows Proactor connect; proxies to 127.0.0.1:9)"
    )
