"""Tests for watchlist I/O: BOM handling, filtering, roundtrip, loud failure.

Every test injects an explicit tmp path. Patching the module global is forbidden:
it is what let these tests write through to the production watchlist.json.
"""

import json
import logging
from pathlib import Path

import pytest

from aer.utils.constants import WATCHLIST_MAX_TICKERS
from aer.utils.watchlist import WatchlistLoadError, load_watchlist, save_watchlist


class TestWatchlistCapBoundary:
    """Boundary tests for the WATCHLIST_MAX_TICKERS cap on load and save."""

    @staticmethod
    def _tickers(n: int) -> list[str]:
        return [f"T{i:03d}" for i in range(n)]

    @pytest.mark.parametrize("n", [WATCHLIST_MAX_TICKERS - 1, WATCHLIST_MAX_TICKERS, WATCHLIST_MAX_TICKERS + 1])
    def test_load_truncates_at_cap(self, tmp_path: Path, n: int) -> None:
        p = tmp_path / "wl.json"
        p.write_text(json.dumps(self._tickers(n)), encoding="utf-8")
        expected = min(n, WATCHLIST_MAX_TICKERS)
        assert len(load_watchlist(p)) == expected

    @pytest.mark.parametrize("n", [WATCHLIST_MAX_TICKERS - 1, WATCHLIST_MAX_TICKERS, WATCHLIST_MAX_TICKERS + 1])
    def test_save_truncates_at_cap(self, tmp_path: Path, n: int) -> None:
        p = tmp_path / "wl.json"
        save_watchlist(self._tickers(n), p)
        expected = min(n, WATCHLIST_MAX_TICKERS)
        assert len(json.loads(p.read_text(encoding="utf-8"))) == expected


class TestLoadWatchlist:
    def test_reads_valid_json_file(self, tmp_path: Path) -> None:
        p = tmp_path / "wl.json"
        p.write_text(json.dumps(["AAPL", "MSFT"]), encoding="utf-8")
        assert load_watchlist(p) == ["AAPL", "MSFT"]

    def test_filters_non_string_entries(self, tmp_path: Path) -> None:
        p = tmp_path / "wl.json"
        p.write_text(json.dumps(["AAPL", 123, None, "MSFT"]), encoding="utf-8")
        assert load_watchlist(p) == ["AAPL", "MSFT"]

    def test_missing_file_raises_and_logs_error(self, tmp_path: Path, caplog) -> None:
        p = tmp_path / "missing.json"
        with caplog.at_level(logging.ERROR, logger="aer.utils.watchlist"):
            with pytest.raises(WatchlistLoadError, match="Unable to load watchlist"):
                load_watchlist(p)
        assert [r for r in caplog.records if r.levelno == logging.ERROR]

    def test_corrupt_json_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "wl.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(WatchlistLoadError, match="Unable to load watchlist"):
            load_watchlist(p)

    def test_bom_prefixed_file_reads(self, tmp_path: Path) -> None:
        p = tmp_path / "wl.json"
        tickers = ["AAPL", "MSFT", "GOOGL"]
        p.write_text(json.dumps(tickers), encoding="utf-8-sig")
        assert load_watchlist(p) == tickers


class TestSaveWatchlist:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "wl.json"
        tickers = ["AAPL", "MSFT", "NVDA"]
        save_watchlist(tickers, p)
        assert load_watchlist(p) == tickers
