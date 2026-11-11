"""Tests for src/data/news_feeds.py.

Covers: Jaccard helpers (deterministic), deduplication logic, aggregator cache hit path,
schema enforcement, and bot-detection fallback behavior.
No live network calls — all external I/O is mocked.
"""

from unittest.mock import MagicMock, patch

from feedparser.exceptions import CharacterEncodingOverride

from aer.data.news_feeds import (
    JACCARD_DEDUP_THRESHOLD,
    JACCARD_RELEVANCE_THRESHOLD,
    _compute_jaccard,
    _deduplicate_headlines,
    _fetch_with_bot_fallback,
    _parse_rss_entries,
    get_news_headlines,
)

# --- _validate_ticker (via public functions) ---


class TestValidateTicker:
    def test_invalid_ticker_returns_empty(self) -> None:
        """Ticker with a space is invalid — get_news_headlines must return []."""
        result = get_news_headlines("AAPL TSLA")
        assert result == []

    def test_too_long_ticker_returns_empty(self) -> None:
        """Ticker longer than 10 characters must return []."""
        result = get_news_headlines("TOOLONGSTOCK")
        assert result == []

    def test_lowercase_ticker_accepted(self) -> None:
        """_validate_ticker uppercases before matching — lowercase input must succeed."""
        cached_result = [
            {
                "title": "Apple earnings",
                "source": "CNBC",
                "url": "",
                "published_ts": "",
                "relevance": 0.5,
                "summary": None,
            }
        ]
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_result
        with patch("aer.data.news_feeds.news_cache", return_value=mock_cache):
            result = get_news_headlines("aapl")
        assert isinstance(result, list)
        assert result == cached_result


# --- _compute_jaccard ---


class TestComputeJaccard:
    def test_constants(self) -> None:
        assert JACCARD_RELEVANCE_THRESHOLD == 0.1
        assert JACCARD_DEDUP_THRESHOLD == 0.6

    def test_identical_sets_returns_one(self) -> None:
        assert _compute_jaccard("aapl apple", "aapl apple") == 1.0

    def test_no_overlap_returns_zero(self) -> None:
        assert _compute_jaccard("tesla tsla earnings", "aapl apple") == 0.0

    def test_empty_text_a_returns_zero(self) -> None:
        assert _compute_jaccard("", "aapl apple") == 0.0

    def test_empty_text_b_returns_zero(self) -> None:
        assert _compute_jaccard("aapl apple", "") == 0.0

    def test_partial_overlap(self) -> None:
        score = _compute_jaccard("apple report", "apple earnings")
        assert 0.0 < score < 1.0

    def test_ticker_in_headline_passes_threshold(self) -> None:
        """Headline mentioning ticker and company should pass relevance filter."""
        score = _compute_jaccard("Apple AAPL reports record quarterly earnings", "AAPL Apple")
        assert score >= JACCARD_RELEVANCE_THRESHOLD

    def test_unrelated_headline_fails_threshold(self) -> None:
        """Headline with zero token overlap should fail relevance filter."""
        score = _compute_jaccard("Crude oil futures rise on supply concerns", "AAPL Apple")
        assert score < JACCARD_RELEVANCE_THRESHOLD

    def test_case_insensitive(self) -> None:
        """Jaccard is case-insensitive (lowercased before tokenizing)."""
        assert _compute_jaccard("AAPL APPLE", "aapl apple") == 1.0


# --- _deduplicate_headlines ---


class TestDeduplicateHeadlines:
    def _make_headline(self, title: str, source: str = "TestSource") -> dict:
        return {
            "title": title,
            "source": source,
            "url": "https://example.com",
            "published_ts": "",
            "relevance": 0.5,
            "summary": None,
        }

    def test_empty_input(self) -> None:
        assert _deduplicate_headlines([]) == []

    def test_single_headline_unchanged(self) -> None:
        h = self._make_headline("Apple AAPL earnings beat estimates")
        result = _deduplicate_headlines([h])
        assert len(result) == 1
        assert result[0]["title"] == h["title"]

    def test_near_duplicate_collapsed(self) -> None:
        """Two headlines with Jaccard >= 0.6 should collapse to one."""
        h1 = self._make_headline("Apple AAPL reports record quarterly earnings beat")
        h2 = self._make_headline("Apple AAPL reports record quarterly earnings")
        result = _deduplicate_headlines([h1, h2])
        assert len(result) == 1

    def test_distinct_headlines_both_kept(self) -> None:
        """Two headlines with Jaccard < 0.6 should both be kept."""
        h1 = self._make_headline("Apple AAPL earnings beat estimates significantly")
        h2 = self._make_headline("Federal Reserve holds interest rates steady after meeting")
        result = _deduplicate_headlines([h1, h2])
        assert len(result) == 2

    def test_first_of_duplicate_pair_is_kept(self) -> None:
        """When collapsing duplicates, the first-seen headline is retained."""
        h1 = self._make_headline("Apple AAPL quarterly earnings beat Wall Street estimates")
        h2 = self._make_headline("Apple AAPL quarterly earnings beat Wall Street")
        result = _deduplicate_headlines([h1, h2])
        assert result[0]["title"] == h1["title"]

    def test_three_headlines_two_near_duplicates(self) -> None:
        """3 headlines where 2 are near-duplicates -> 2 retained."""
        h1 = self._make_headline("Apple AAPL earnings beat estimates significantly")
        h2 = self._make_headline("Apple AAPL earnings beat estimates")  # near-dup of h1
        h3 = self._make_headline("Fed holds rates steady -- statement released")
        result = _deduplicate_headlines([h1, h2, h3])
        assert len(result) == 2


# --- get_news_headlines aggregator (mocked) ---

_FAKE_HEADLINES: list[dict] = [
    {
        "title": "Apple AAPL quarterly earnings beat Wall Street estimates",
        "source": "CNBC",
        "url": "https://cnbc.com/story/aapl-earnings",
        "published_ts": "2026-05-20T14:00:00+00:00",
        "summary": "Apple reported strong quarterly results.",
    },
    {
        "title": "Federal Reserve holds rates steady",
        "source": "BBC",
        "url": "https://bbc.co.uk/news/fed-rates",
        "published_ts": "2026-05-20T12:00:00+00:00",
        "summary": None,
    },
]


class TestGetNewsHeadlines:
    def test_cache_hit_returns_cached_value(self) -> None:
        """get_news_headlines() returns cached list without fetching sources."""
        cached_result = [{**_FAKE_HEADLINES[0], "relevance": 0.5}]
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_result

        with patch("aer.data.news_feeds.news_cache", return_value=mock_cache):
            result = get_news_headlines("AAPL")

        assert result == cached_result
        mock_cache.set.assert_not_called()

    def test_result_schema_all_six_fields(self) -> None:
        """Every returned headline must have all six fields including relevance."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # cache miss

        mock_meta = {"name": "Apple Inc.", "ticker": "AAPL", "sector": "Technology"}

        aapl_headline = {
            "title": "Apple AAPL earnings surge on iPhone demand",
            "source": "CNBC",
            "url": "https://cnbc.com/aapl",
            "published_ts": "2026-05-20T14:00:00+00:00",
            "summary": "Apple beat estimates.",
        }

        with (
            patch("aer.data.news_feeds.news_cache", return_value=mock_cache),
            patch("aer.data.news_feeds.get_company_metadata", return_value=mock_meta),
            patch("aer.data.news_feeds._fetch_cnbc_rss", return_value=[aapl_headline]),
            patch("aer.data.news_feeds._fetch_bbc_rss", return_value=[]),
            patch("aer.data.news_feeds._fetch_marketwatch_rss", return_value=[]),
            patch("aer.data.news_feeds._fetch_barchart", return_value=[]),
            patch("aer.data.news_feeds._fetch_finviz_news", return_value=[]),
            patch("aer.data.news_feeds._fetch_zacks", return_value=[]),
            patch("aer.data.news_feeds._fetch_seekingalpha", return_value=[]),
            patch("aer.data.news_feeds._fetch_investing_com", return_value=[]),
        ):
            result = get_news_headlines("AAPL")

        required_keys = {"title", "source", "url", "published_ts", "relevance", "summary"}
        for h in result:
            assert required_keys.issubset(h.keys()), f"Missing keys in: {h}"

    def test_relevance_filter_removes_unrelated(self) -> None:
        """Headlines with Jaccard < 0.1 vs 'AAPL Apple Inc.' must not appear in result."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        mock_meta = {"name": "Apple Inc.", "ticker": "AAPL", "sector": "Technology"}

        unrelated = {
            "title": "Crude oil futures rise on OPEC supply decisions",
            "source": "BBC",
            "url": "https://bbc.co.uk/oil",
            "published_ts": "",
            "summary": None,
        }

        with (
            patch("aer.data.news_feeds.news_cache", return_value=mock_cache),
            patch("aer.data.news_feeds.get_company_metadata", return_value=mock_meta),
            patch("aer.data.news_feeds._fetch_cnbc_rss", return_value=[]),
            patch("aer.data.news_feeds._fetch_bbc_rss", return_value=[unrelated]),
            patch("aer.data.news_feeds._fetch_marketwatch_rss", return_value=[]),
            patch("aer.data.news_feeds._fetch_barchart", return_value=[]),
            patch("aer.data.news_feeds._fetch_finviz_news", return_value=[]),
            patch("aer.data.news_feeds._fetch_zacks", return_value=[]),
            patch("aer.data.news_feeds._fetch_seekingalpha", return_value=[]),
            patch("aer.data.news_feeds._fetch_investing_com", return_value=[]),
        ):
            result = get_news_headlines("AAPL")

        assert result == []

    def test_all_returned_headlines_meet_relevance_threshold(self) -> None:
        """Every headline in the result must have relevance >= JACCARD_RELEVANCE_THRESHOLD."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        mock_meta = {"name": "Apple Inc.", "ticker": "AAPL", "sector": "Technology"}

        headlines = [
            {
                "title": "Apple AAPL earnings surge on iPhone demand record quarter",
                "source": "CNBC",
                "url": "https://cnbc.com/aapl",
                "published_ts": "2026-05-20T14:00:00+00:00",
                "summary": None,
            },
            {
                "title": "Random unrelated headline about shipping containers",
                "source": "BBC",
                "url": "https://bbc.co.uk/shipping",
                "published_ts": "",
                "summary": None,
            },
        ]

        with (
            patch("aer.data.news_feeds.news_cache", return_value=mock_cache),
            patch("aer.data.news_feeds.get_company_metadata", return_value=mock_meta),
            patch("aer.data.news_feeds._fetch_cnbc_rss", return_value=[headlines[0]]),
            patch("aer.data.news_feeds._fetch_bbc_rss", return_value=[headlines[1]]),
            patch("aer.data.news_feeds._fetch_marketwatch_rss", return_value=[]),
            patch("aer.data.news_feeds._fetch_barchart", return_value=[]),
            patch("aer.data.news_feeds._fetch_finviz_news", return_value=[]),
            patch("aer.data.news_feeds._fetch_zacks", return_value=[]),
            patch("aer.data.news_feeds._fetch_seekingalpha", return_value=[]),
            patch("aer.data.news_feeds._fetch_investing_com", return_value=[]),
        ):
            result = get_news_headlines("AAPL")

        for h in result:
            assert (
                h["relevance"] >= JACCARD_RELEVANCE_THRESHOLD
            ), f"headline below threshold: {h['title']} (relevance={h['relevance']})"

    def test_cache_write_called_on_miss(self) -> None:
        """On cache miss, data_cache().set() must be called with ttl=TTL_NEWS."""
        from aer.data.news_feeds import TTL_NEWS  # should be importable (re-exported from utils)

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        mock_meta = {"name": "Apple Inc.", "ticker": "AAPL", "sector": "Technology"}

        with (
            patch("aer.data.news_feeds.news_cache", return_value=mock_cache),
            patch("aer.data.news_feeds.get_company_metadata", return_value=mock_meta),
            patch("aer.data.news_feeds._fetch_cnbc_rss", return_value=[]),
            patch("aer.data.news_feeds._fetch_bbc_rss", return_value=[]),
            patch("aer.data.news_feeds._fetch_marketwatch_rss", return_value=[]),
            patch("aer.data.news_feeds._fetch_barchart", return_value=[]),
            patch("aer.data.news_feeds._fetch_finviz_news", return_value=[]),
            patch("aer.data.news_feeds._fetch_zacks", return_value=[]),
            patch("aer.data.news_feeds._fetch_seekingalpha", return_value=[]),
            patch("aer.data.news_feeds._fetch_investing_com", return_value=[]),
        ):
            get_news_headlines("AAPL")

        mock_cache.set.assert_called_once()
        call_kwargs = mock_cache.set.call_args.kwargs
        assert (
            call_kwargs.get("ttl") == TTL_NEWS
        ), f"cache.set() must be called with ttl=TTL_NEWS, got ttl={call_kwargs.get('ttl')}"
        call_args_positional = mock_cache.set.call_args.args
        assert call_args_positional[0].startswith("news:AAPL:"), f"unexpected cache key: {call_args_positional[0]}"
        assert isinstance(call_args_positional[1], list)


# --- _parse_rss_entries ---


class TestParseRssEntries:
    def test_bozo_encoding_override_returns_entries(self) -> None:
        """CharacterEncodingOverride is benign — entries must still be returned."""
        entry = MagicMock()
        entry.get = lambda k, d="": {"title": "Apple earnings", "link": "https://x.com"}.get(k, d)
        entry.published_parsed = None

        mock_d = MagicMock()
        mock_d.bozo = True
        mock_d.bozo_exception = CharacterEncodingOverride()
        mock_d.entries = [entry]

        with patch("aer.data.news_feeds.feedparser.parse", return_value=mock_d):
            result = _parse_rss_entries("https://example.com", "TestSource")
        assert len(result) == 1

    def test_bozo_fatal_error_returns_empty(self) -> None:
        """Fatal bozo errors (malformed XML etc.) must return []."""
        mock_d = MagicMock()
        mock_d.bozo = True
        mock_d.bozo_exception = Exception("Malformed XML")

        with patch("aer.data.news_feeds.feedparser.parse", return_value=mock_d):
            result = _parse_rss_entries("https://example.com", "TestSource")
        assert result == []


# --- Bot-detection fallback ---


class TestBotDetectionFallback:
    def test_403_returns_empty_list(self) -> None:
        """HTTP 403 response returns [] silently (no exception raised)."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Access denied"

        with patch("aer.data.news_feeds.requests.get", return_value=mock_response):
            result = _fetch_with_bot_fallback("https://example.com", "TestSource")

        assert result == []

    def test_incapsula_short_response_returns_empty_list(self) -> None:
        """HTTP 200 with short body (Incapsula 965-byte page) returns [] silently."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "x" * 100  # shorter than _MIN_REAL_CONTENT_LENGTH (5000)

        with patch("aer.data.news_feeds.requests.get", return_value=mock_response):
            result = _fetch_with_bot_fallback("https://zacks.com/AAPL", "Zacks")

        assert result == []

    def test_network_exception_returns_empty_list(self) -> None:
        """Network exception (timeout, DNS error) returns [] without raising."""
        with patch("aer.data.news_feeds.requests.get", side_effect=OSError("Network unreachable")):
            result = _fetch_with_bot_fallback("https://seekingalpha.com/AAPL", "SeekingAlpha")

        assert result == []
