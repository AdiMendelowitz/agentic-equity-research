"""Multi-source sentiment feeds: ApeWisdom (Reddit), Stocktwits, FinViz news, Yahoo Finance news."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from aer.config import get_settings

# Re-exports for backward compatibility - functions migrated to src/aer/data/news_feeds.py
from aer.data.news_feeds import get_finviz_news, get_yahoo_news  # noqa: F401
from aer.utils.cache import TTL_ANALYSIS, agent_cache
from aer.utils.vintage_tracker import record_vintage

logger = logging.getLogger(__name__)

_AW_PAGES = 4
_SENTIMENT_WALL_CLOCK_DEADLINE_SECONDS: float = 20.0  # 4 sources × 10s per-request worst case with margin


def _headers() -> dict[str, str]:
    return {"User-Agent": get_settings().reddit_user_agent}


def _fetch_aw_page(page: int) -> list[dict]:
    resp = requests.get(
        f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}",
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def get_apewisdom(ticker: str) -> dict[str, Any]:
    """Return Reddit mention stats for ticker from ApeWisdom. Cached TTL_ANALYSIS."""
    ticker = ticker.upper()
    key = f"sentiment:apewisdom:{ticker}"
    cache = agent_cache()

    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    not_found: dict[str, Any] = {
        "ticker": ticker,
        "found": False,
        "rank": None,
        "mentions_24h": None,
        "upvotes_24h": None,
        "rank_change_24h": None,
    }

    try:
        found: dict[str, Any] | None = None
        with ThreadPoolExecutor(max_workers=_AW_PAGES) as executor:
            futures = {executor.submit(_fetch_aw_page, page): page for page in range(1, _AW_PAGES + 1)}
            for future in as_completed(futures):
                try:
                    for item in future.result():
                        if item.get("ticker", "").upper() == ticker:
                            rank_now = item.get("rank")
                            rank_ago = item.get("rank_24h_ago")
                            found = {
                                **not_found,
                                "found": True,
                                "rank": rank_now,
                                "mentions_24h": item.get("mentions"),
                                "upvotes_24h": item.get("upvotes"),
                                "rank_change_24h": (rank_ago - rank_now) if rank_now and rank_ago else None,
                            }
                            break
                except Exception as e:
                    logger.warning("ApeWisdom page fetch failed: %s", e)
                if found:
                    # cancel() is a no-op for already-running futures - all pages start immediately
                    # with max_workers=_AW_PAGES. All pages complete regardless; ApeWisdom API
                    # latency is low enough this is acceptable.
                    break

        if found:
            cache.set(key, found, ttl=TTL_ANALYSIS)
            return found
        cache.set(key, not_found, ttl=TTL_ANALYSIS)
        return not_found

    except Exception as e:
        logger.warning("ApeWisdom request failed for %s: %s", ticker, e)
        return not_found


def get_stocktwits(ticker: str, limit: int = 30) -> dict[str, Any]:
    """Return bullish/bearish breakdown from Stocktwits. Cached TTL_ANALYSIS."""
    ticker = ticker.upper()
    key = f"sentiment:stocktwits:{ticker}:{limit}"
    cache = agent_cache()

    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    empty: dict[str, Any] = {
        "ticker": ticker,
        "message_count": 0,
        "bullish_count": 0,
        "bearish_count": 0,
        "neutral_count": 0,
        "bullish_pct": None,
        "bearish_pct": None,
        "sample_messages": [],
    }

    try:
        resp = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            headers=_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        messages = resp.json().get("messages", [])[:limit]

        bullish = bearish = 0
        samples: list[str] = []

        for msg in messages:
            body = msg.get("body", "")
            signal = ((msg.get("entities") or {}).get("sentiment") or {}).get("basic", "").lower()
            if signal == "bullish":
                bullish += 1
            elif signal == "bearish":
                bearish += 1
            if body and len(samples) < 5:
                samples.append(body[:200])

        total = len(messages)
        result = {
            "ticker": ticker,
            "message_count": total,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": total - bullish - bearish,
            "bullish_pct": round(bullish / total, 3) if total else None,
            "bearish_pct": round(bearish / total, 3) if total else None,
            "sample_messages": samples,
        }
        cache.set(key, result, ttl=TTL_ANALYSIS)
        return result

    except Exception as e:
        logger.warning("Stocktwits request failed for %s: %s", ticker, e)
        return empty


def get_cboe_pc_ratio() -> dict[str, Any]:
    """CBOE equity put/call ratio. Cached TTL_ANALYSIS.

    ^PCCE and ^PCCX were removed from Yahoo Finance. Returns null fields until
    a replacement data source is wired up.
    """
    key = "sentiment:cboe_pc_ratio"
    cache = agent_cache()

    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    result: dict[str, Any] = {
        "equity_pc_ratio": None,
        "total_pc_ratio": None,
        "equity_pc_5d_avg": None,
        "sentiment_label": None,
    }
    cache.set(key, result, ttl=TTL_ANALYSIS)
    return result


def get_sentiment_snapshot(ticker: str) -> dict[str, Any]:
    """Fetch all sources. FinViz is primary news; Yahoo Finance is fallback."""
    ticker = ticker.upper()

    def _aw() -> dict[str, Any]:
        return get_apewisdom(ticker)

    def _st() -> dict[str, Any]:
        return get_stocktwits(ticker)

    def _finviz() -> dict[str, Any]:
        return get_finviz_news(ticker)

    def _yahoo() -> dict[str, Any]:
        return get_yahoo_news(ticker)

    _empty_aw: dict[str, Any] = {
        "ticker": ticker,
        "found": False,
        "rank": None,
        "mentions_24h": None,
        "upvotes_24h": None,
        "rank_change_24h": None,
    }
    _empty_st: dict[str, Any] = {
        "ticker": ticker,
        "message_count": 0,
        "bullish_count": 0,
        "bearish_count": 0,
        "neutral_count": 0,
        "bullish_pct": None,
        "bearish_pct": None,
        "sample_messages": [],
    }
    _empty_finviz: dict[str, Any] = {"ticker": ticker, "article_count": 0, "headlines": [], "source": "finviz"}
    _empty_yahoo: dict[str, Any] = {"ticker": ticker, "article_count": 0, "headlines": []}

    executor = ThreadPoolExecutor(max_workers=4)
    try:
        fut_aw = executor.submit(_aw)
        fut_st = executor.submit(_st)
        fut_finviz = executor.submit(_finviz)
        fut_yahoo = executor.submit(_yahoo)

        # Use a global wall-clock deadline so sequential .result() calls
        # cannot stack to 4×15 = 60 seconds in the pathological case.
        deadline = time.monotonic() + _SENTIMENT_WALL_CLOCK_DEADLINE_SECONDS
        futures_and_defaults = [
            (fut_aw, _empty_aw),
            (fut_st, _empty_st),
            (fut_finviz, _empty_finviz),
            (fut_yahoo, _empty_yahoo),
        ]
        results = []
        for fut, default in futures_and_defaults:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                results.append(fut.result(timeout=remaining) if remaining > 0 else default)
            except Exception as exc:
                logger.warning("Sentiment feed failed: %s", exc)
                results.append(default)
        aw_result, st_result, finviz_result, yahoo_result = results
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    finviz_ok = finviz_result.get("article_count", 0) > 0

    snapshot: dict[str, Any] = {
        "apewisdom": aw_result,
        "stocktwits": st_result,
        "finviz_news": finviz_result,
        "yahoo_news": yahoo_result,
        "news_primary_source": "finviz" if finviz_ok else "yahoo",
    }
    record_vintage("sentiment")
    return snapshot
