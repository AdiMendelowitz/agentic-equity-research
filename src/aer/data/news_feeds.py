"""Multi-source news feeds: CNBC RSS, BBC RSS, MarketWatch RSS, Barchart HTML,
Finviz via finvizfinance library, SeekingAlpha (bot-fallback), Investing.com (bot-fallback),
Zacks (bot-fallback - Incapsula-blocked), Yahoo Finance (legacy re-export support).

Zero LLM calls - pure Python data layer.
get_news_headlines() is the single public entry point.
"""

import calendar
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from feedparser.exceptions import CharacterEncodingOverride

from aer.data.prices import get_company_metadata
from aer.utils.cache import TTL_NEWS, news_cache

logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")


def _validate_ticker(ticker: str) -> str:
    t = ticker.upper()
    if not _TICKER_RE.match(t):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    return t


# --- Jaccard thresholds (fixed; do not change) ---
JACCARD_RELEVANCE_THRESHOLD: float = 0.1
JACCARD_DEDUP_THRESHOLD: float = 0.6

# --- HTTP constants ---
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_REQUEST_TIMEOUT = 10
_BOT_DETECTION_CODES: frozenset[int] = frozenset({401, 403, 429, 503})
_MIN_REAL_CONTENT_LENGTH = 5000  # Incapsula pages are ~965 bytes; real pages always >5000

# --- RSS feed URLs (verified live 2026-05-20) ---
_CNBC_RSS_URLS = [
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
]
_BBC_RSS_URL = "https://feeds.bbci.co.uk/news/business/rss.xml"
_MARKETWATCH_RSS_URLS = [
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",  # Reuters RSS DNS-dead; MarketWatch substitutes
    "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
]

# --- HTML scraping URL templates ---
_BARCHART_BASE = "https://www.barchart.com"
_BARCHART_URL_TEMPLATE = "https://www.barchart.com/stocks/quotes/{ticker}"
_BARCHART_NEWS_SELECTOR = 'a[href*="/story/news/"]'
_BARCHART_MIN_TEXT_LENGTH = 20  # filter out nav links shorter than this

# --- Tier C (bot-blocked) URL templates ---
_ZACKS_URL_TEMPLATE = "https://www.zacks.com/stock/news/{ticker}"
_SEEKINGALPHA_URL_TEMPLATE = "https://seekingalpha.com/symbol/{ticker}/news"
_INVESTING_COM_URL_TEMPLATE = "https://www.investing.com/search/?q={ticker}&tab=news"

# --- Executor config ---
_EXECUTOR_MAX_WORKERS = 8
_WALL_CLOCK_DEADLINE_SECONDS = 25.0


# ---------------------------------------------------------------------------
# Jaccard helpers (module-level for testability)
# ---------------------------------------------------------------------------


def _compute_jaccard(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    union = len(tokens_a | tokens_b)
    return len(tokens_a & tokens_b) / union if union else 0.0


def _deduplicate_headlines(headlines: list[dict]) -> list[dict]:
    """Collapse headlines where title Jaccard >= JACCARD_DEDUP_THRESHOLD to single entry."""
    kept: list[dict] = []
    for candidate in headlines:
        duplicate = False
        for retained in kept:
            if _compute_jaccard(candidate["title"], retained["title"]) >= JACCARD_DEDUP_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# RSS helper (shared by Tier A fetchers)
# ---------------------------------------------------------------------------


def _parse_rss_entries(url: str, source_name: str) -> list[dict]:
    """Fetch and parse one RSS feed URL. Returns [] on any failure."""
    try:
        d = feedparser.parse(url)
        if d.bozo and not isinstance(d.bozo_exception, CharacterEncodingOverride):
            logger.warning("RSS parse error for %s (%s): %s", source_name, url, d.bozo_exception)
            return []
        results = []
        for entry in d.entries:
            pp = entry.get("published_parsed")
            published_ts = datetime.fromtimestamp(calendar.timegm(pp), tz=UTC).isoformat() if pp else ""
            title = entry.get("title", "").strip()
            if not title:
                continue
            results.append(
                {
                    "title": title,
                    "source": source_name,
                    "url": entry.get("link", ""),
                    "published_ts": published_ts,
                    "summary": entry.get("summary") or None,
                }
            )
        return results
    except Exception as exc:
        logger.warning("RSS fetch failed for %s (%s): %s", source_name, url, exc)
        return []


# ---------------------------------------------------------------------------
# Tier A - RSS sources (expected to work; warning on failure)
# ---------------------------------------------------------------------------


def _fetch_cnbc_rss() -> list[dict]:
    results = []
    for url in _CNBC_RSS_URLS:
        results.extend(_parse_rss_entries(url, "CNBC"))
    return results


def _fetch_bbc_rss() -> list[dict]:
    return _parse_rss_entries(_BBC_RSS_URL, "BBC")


def _fetch_marketwatch_rss() -> list[dict]:
    results = []
    for url in _MARKETWATCH_RSS_URLS:
        results.extend(_parse_rss_entries(url, "MarketWatch RSS"))
    return results


# ---------------------------------------------------------------------------
# Tier B - HTML scraping (expected to work; warning on failure)
# ---------------------------------------------------------------------------


def _fetch_barchart(ticker: str) -> list[dict]:
    try:
        url = _BARCHART_URL_TEMPLATE.format(ticker=ticker.upper())
        r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        links = [
            a for a in soup.select(_BARCHART_NEWS_SELECTOR) if len(a.get_text(strip=True)) > _BARCHART_MIN_TEXT_LENGTH
        ]
        seen_urls: set[str] = set()
        results = []
        for a in links:
            href = a.get("href", "")
            if href in seen_urls:
                continue
            seen_urls.add(href)
            results.append(
                {
                    "title": a.get_text(strip=True),
                    "source": "Barchart",
                    "url": urljoin(_BARCHART_BASE, href),
                    "published_ts": "",
                    "summary": None,
                }
            )
        return results
    except Exception as exc:
        logger.warning("Barchart scrape failed for %s: %s", ticker, exc)
        return []


def _fetch_finviz_news(ticker: str) -> list[dict]:
    try:
        from finvizfinance.quote import finvizfinance as fvz  # noqa: PLC0415

        df = fvz(ticker.upper()).ticker_news()
        if df is None or df.empty:
            return []
        results = []
        for _, row in df.iterrows():
            title = str(row.get("Title", "")).strip()
            if not title or title == "nan":
                continue
            dt = row.get("Date")
            published_ts = dt.isoformat() if hasattr(dt, "isoformat") else ""
            results.append(
                {
                    "title": title,
                    "source": "Finviz",
                    "url": str(row.get("Link", "")),
                    "published_ts": published_ts,
                    "summary": None,
                }
            )
        return results
    except Exception as exc:
        logger.warning("Finviz news failed for %s: %s", ticker, exc)
        return []


# ---------------------------------------------------------------------------
# Tier C - bot-detection fallback (debug-only logging; always returns [])
# ---------------------------------------------------------------------------


def _fetch_with_bot_fallback(url: str, source_name: str) -> list[dict]:
    """Attempt HTTP fetch; return [] silently on bot-detection block or Incapsula page."""
    try:
        r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_REQUEST_TIMEOUT)
        if r.status_code in _BOT_DETECTION_CODES:
            logger.debug("%s blocked (status=%d)", source_name, r.status_code)
            return []
        if len(r.text) < _MIN_REAL_CONTENT_LENGTH:
            logger.debug("%s short response (len=%d) -- likely bot-detection page", source_name, len(r.text))
            return []
        logger.debug("content received but no parser implemented for %s", source_name)
        return []
    except Exception as exc:
        logger.debug("%s fetch failed: %s", source_name, exc)
        return []


def _fetch_zacks(ticker: str) -> list[dict]:
    return _fetch_with_bot_fallback(_ZACKS_URL_TEMPLATE.format(ticker=ticker.upper()), "Zacks")


def _fetch_seekingalpha(ticker: str) -> list[dict]:
    return _fetch_with_bot_fallback(_SEEKINGALPHA_URL_TEMPLATE.format(ticker=ticker.upper()), "SeekingAlpha")


def _fetch_investing_com(ticker: str) -> list[dict]:
    return _fetch_with_bot_fallback(_INVESTING_COM_URL_TEMPLATE.format(ticker=ticker.upper()), "Investing.com")


# ---------------------------------------------------------------------------
# Legacy compatibility (migrated from sentiment_feeds.py - sentinel agent unchanged)
# ---------------------------------------------------------------------------


def get_finviz_news(ticker: str, limit: int = 15) -> dict[str, Any]:
    """Return recent news headlines from FinViz in legacy sentiment_feeds format. Cached via news_cache (TTL_NEWS)."""
    try:
        ticker = _validate_ticker(ticker)
    except ValueError:
        logger.warning("get_finviz_news called with invalid ticker: %r", ticker)
        return {"ticker": ticker, "article_count": 0, "headlines": [], "source": "finviz"}
    key = f"finviz_news:{ticker}:{limit}"
    cache = news_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    raw = _fetch_finviz_news(ticker)[:limit]
    headlines = [{"title": h["title"], "publisher": "FinViz", "url": h["url"]} for h in raw]
    result: dict[str, Any] = {
        "ticker": ticker,
        "article_count": len(headlines),
        "headlines": headlines,
        "source": "finviz",
    }
    cache.set(key, result, ttl=TTL_NEWS)
    return result


def get_yahoo_news(ticker: str, limit: int = 10) -> dict[str, Any]:
    """Return recent news headlines from Yahoo Finance in legacy sentiment_feeds format.

    Cached via news_cache (TTL_NEWS).
    """
    try:
        ticker = _validate_ticker(ticker)
    except ValueError:
        logger.warning("get_yahoo_news called with invalid ticker: %r", ticker)
        return {"ticker": ticker, "article_count": 0, "headlines": []}
    import yfinance as yf  # noqa: PLC0415

    key = f"yahoo_news:{ticker}:{limit}"
    cache = news_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    empty: dict[str, Any] = {"ticker": ticker, "article_count": 0, "headlines": []}
    try:
        news = yf.Ticker(ticker).news or []
        headlines = []
        for article in news[:limit]:
            content = article.get("content", {})
            title = (content.get("title", "") if isinstance(content, dict) else "") or article.get("title", "")
            publisher = ""
            if isinstance(content, dict):
                provider = content.get("provider", {})
                publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
            if title:
                headlines.append({"title": title, "publisher": publisher})
        result: dict[str, Any] = {"ticker": ticker, "article_count": len(headlines), "headlines": headlines}
        cache.set(key, result, ttl=TTL_NEWS)
        return result
    except Exception as exc:
        logger.warning("Yahoo Finance news failed for %s: %s", ticker, exc)
        return empty


# ---------------------------------------------------------------------------
# Public aggregator - single entry point for consumers
# ---------------------------------------------------------------------------


def get_news_headlines(ticker: str, *, as_of: date | None = None) -> list[dict]:
    """Fetch, score, dedup, and cache headlines for ticker from all configured sources.

    as_of: run-level date for the cache key (prevents midnight-crossing
    key mismatch between prefetch write and downstream probe). Defaults to today (UTC).
    Returns list[dict] per schema with relevance injected. Empty list if all sources fail.
    Cache key: news:{ticker}:{date_str}, TTL: TTL_NEWS (3600s).
    """
    try:
        ticker = _validate_ticker(ticker)
    except ValueError:
        logger.warning("get_news_headlines called with invalid ticker: %r", ticker)
        return []
    stamp = as_of if as_of is not None else datetime.now(tz=UTC).date()
    date_str = stamp.isoformat()
    key = f"news:{ticker}:{date_str}"
    cache = news_cache()

    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    meta = get_company_metadata(ticker)
    company_name: str = meta.get("name", ticker)
    # WARNING: Jaccard relevance at threshold=0.1 is calibrated for large-cap tickers
    # with distinctive names (AAPL/Apple, TSLA/Tesla). For tickers whose symbols are
    # common English words (ON, IT, GO, CAR, CAT, RE), false positive rate is high.
    # Consider a higher threshold (0.15-0.2) for single-character/common-word tickers,
    # or add a phrase-match pre-filter requiring the ticker symbol to appear verbatim.
    jaccard_query = f"{ticker} {company_name}"

    executor = ThreadPoolExecutor(max_workers=_EXECUTOR_MAX_WORKERS)
    try:
        futures = [
            executor.submit(_fetch_cnbc_rss),
            executor.submit(_fetch_bbc_rss),
            executor.submit(_fetch_marketwatch_rss),
            executor.submit(_fetch_barchart, ticker),
            executor.submit(_fetch_finviz_news, ticker),
            executor.submit(_fetch_zacks, ticker),
            executor.submit(_fetch_seekingalpha, ticker),
            executor.submit(_fetch_investing_com, ticker),
        ]
        deadline = time.monotonic() + _WALL_CLOCK_DEADLINE_SECONDS
        raw_lists: list[list[dict]] = []
        for fut in futures:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                raw_lists.append(fut.result(timeout=remaining) if remaining > 0 else [])
            except Exception as exc:
                logger.warning("News source fetch failed: %s", exc)
                raw_lists.append([])
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    all_headlines: list[dict] = []
    for lst in raw_lists:
        all_headlines.extend(lst)

    scored: list[dict] = []
    for h in all_headlines:
        score = _compute_jaccard(h["title"], jaccard_query)
        if score >= JACCARD_RELEVANCE_THRESHOLD:
            scored.append({**h, "relevance": score})

    deduped = _deduplicate_headlines(scored)
    result = sorted(deduped, key=lambda h: h["relevance"], reverse=True)

    cache.set(key, result, ttl=TTL_NEWS)
    return result


__all__ = [
    "get_news_headlines",
    "get_finviz_news",
    "get_yahoo_news",
    "JACCARD_RELEVANCE_THRESHOLD",
    "JACCARD_DEDUP_THRESHOLD",
    "TTL_NEWS",
]
