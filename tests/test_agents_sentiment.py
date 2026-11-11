"""Tests for sentiment analysis agent: _score_confidence, build_sentiment_prompt, run()."""

from unittest.mock import MagicMock, patch

import pytest

from aer.data.cache import TTL_ANALYSIS, Cache
from aer.utils.cache import _DomainCache


@pytest.fixture
def fresh_cache(tmp_path) -> Cache:
    return Cache(db_path=str(tmp_path / "sentiment.db"))


@pytest.fixture
def fresh_domain_cache(fresh_cache: Cache) -> _DomainCache:
    """_DomainCache wrapping fresh_cache with agent domain — mirrors agent_cache() singleton."""
    return _DomainCache(fresh_cache, TTL_ANALYSIS, "agent")


def _snap(aw_found: bool = True, st_count: int = 10, yn_count: int = 5) -> dict:
    return {
        "apewisdom": {
            "found": aw_found,
            "rank": 5,
            "rank_change_24h": 3,
            "mentions_24h": 1000,
            "upvotes_24h": 500,
        },
        "stocktwits": {
            "message_count": st_count,
            "bullish_count": 6,
            "bearish_count": 3,
            "neutral_count": 1,
            "bullish_pct": 0.6,
            "bearish_pct": 0.3,
            "sample_messages": ["bullish on AAPL"],
        },
        "yahoo_news": {
            "article_count": yn_count,
            "headlines": [{"title": "Apple surges", "publisher": "Reuters"}],
        },
        "finviz_news": {"article_count": 0, "headlines": [], "ticker": "AAPL"},
        "news_primary_source": "yahoo",
    }


class TestScoreConfidence:
    def test_all_three_sources_active(self) -> None:
        from aer.agents.sentiment import _score_confidence

        conf = _score_confidence(_snap(aw_found=True, st_count=10, yn_count=5))
        assert conf == pytest.approx(0.85, abs=0.01)

    def test_no_sources_active(self) -> None:
        from aer.agents.sentiment import _score_confidence

        conf = _score_confidence(_snap(aw_found=False, st_count=0, yn_count=0))
        assert conf == pytest.approx(0.2, abs=0.01)

    def test_one_source_active(self) -> None:
        from aer.agents.sentiment import _score_confidence

        conf = _score_confidence(_snap(aw_found=True, st_count=0, yn_count=0))
        expected = 0.2 + (1 / 3) * 0.65
        assert conf == pytest.approx(expected, abs=0.01)

    def test_two_sources_active(self) -> None:
        from aer.agents.sentiment import _score_confidence

        conf = _score_confidence(_snap(aw_found=True, st_count=10, yn_count=0))
        expected = 0.2 + (2 / 3) * 0.65
        assert conf == pytest.approx(expected, abs=0.01)

    def test_all_sources_exact_value(self) -> None:
        from aer.agents.sentiment import _score_confidence

        # 3 sources active: 0.2 + (3/3) * 0.65 = 0.85
        assert _score_confidence(_snap(aw_found=True, st_count=10, yn_count=5)) == pytest.approx(0.85)


class TestBuildPrompt:
    def test_contains_ticker(self) -> None:
        from aer.utils.prompt_builders import build_sentiment_prompt

        _, prompt = build_sentiment_prompt("AAPL", _snap(), "MIXED")
        assert "AAPL" in prompt

    def test_apewisdom_rank_included(self) -> None:
        from aer.utils.prompt_builders import build_sentiment_prompt

        _, prompt = build_sentiment_prompt("AAPL", _snap(aw_found=True), "MIXED")
        assert "#5" in prompt

    def test_apewisdom_not_found_message(self) -> None:
        from aer.utils.prompt_builders import build_sentiment_prompt

        _, prompt = build_sentiment_prompt("AAPL", _snap(aw_found=False), "MIXED")
        assert "low visibility" in prompt

    def test_no_stocktwits_message(self) -> None:
        from aer.utils.prompt_builders import build_sentiment_prompt

        _, prompt = build_sentiment_prompt("AAPL", _snap(st_count=0), "MIXED")
        assert "No Stocktwits data" in prompt

    def test_no_yahoo_news_message(self) -> None:
        from aer.utils.prompt_builders import build_sentiment_prompt

        _, prompt = build_sentiment_prompt("AAPL", _snap(yn_count=0), "MIXED")
        assert "No recent news" in prompt

    def test_rank_change_rising(self) -> None:
        from aer.utils.prompt_builders import build_sentiment_prompt

        snap = _snap()
        snap["apewisdom"]["rank_change_24h"] = 5
        _, prompt = build_sentiment_prompt("AAPL", snap, "MIXED")
        assert "rising" in prompt

    def test_rank_change_falling(self) -> None:
        from aer.utils.prompt_builders import build_sentiment_prompt

        snap = _snap()
        snap["apewisdom"]["rank_change_24h"] = -3
        _, prompt = build_sentiment_prompt("AAPL", snap, "MIXED")
        assert "falling" in prompt


class TestSentimentRun:
    def _run(self, domain_cache: _DomainCache, snap: dict | None = None, ticker: str = "AAPL") -> object:
        if snap is None:
            snap = _snap()
        mock_response = MagicMock()
        mock_response.content = "Mixed sentiment."
        with patch("aer.agents.sentiment.agent_cache", return_value=domain_cache):
            with patch("aer.agents.sentiment.get_sentiment_snapshot", return_value=snap):
                with patch("aer.agents.sentiment.complete") as mock_complete:
                    mock_complete.return_value = mock_response
                    from aer.agents.sentiment import run

                    return run(ticker)

    def test_returns_agent_output(self, fresh_domain_cache: _DomainCache) -> None:
        from aer.models import AgentOutput

        result = self._run(fresh_domain_cache)
        assert isinstance(result, AgentOutput)
        assert result.agent_name == "sentiment"

    def test_no_data_returns_low_confidence_output(self, fresh_domain_cache: _DomainCache) -> None:
        snap = _snap(aw_found=False, st_count=0, yn_count=0)
        with patch("aer.agents.sentiment.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.sentiment.get_sentiment_snapshot", return_value=snap):
                from aer.agents.sentiment import run

                result = run("AAPL")
        assert result.confidence == 0.0
        assert "No sentiment data" in result.content

    def test_cache_hit_skips_llm(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        from aer.models import AgentOutput

        cached = AgentOutput(agent_name="sentiment", content="cached text", confidence=0.7)
        fresh_domain_cache.set("agent:sentiment:AAPL", cached.model_dump(mode="json"), ttl=3600)
        with patch("aer.agents.sentiment.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.sentiment.complete") as mock_complete:
                from aer.agents.sentiment import run

                result = run("AAPL")
        mock_complete.assert_not_called()
        assert result.content == "cached text"

    def test_ticker_uppercased(self, fresh_domain_cache: _DomainCache) -> None:
        mock_response = MagicMock()
        mock_response.content = "text"
        with patch("aer.agents.sentiment.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.sentiment.get_sentiment_snapshot", return_value=_snap()) as mock_snap:
                with patch("aer.agents.sentiment.complete") as mock_complete:
                    mock_complete.return_value = mock_response
                    from aer.agents.sentiment import run

                    run("aapl")
        mock_snap.assert_called_once_with("AAPL")

    def test_sources_include_all_three(self, fresh_domain_cache: _DomainCache) -> None:
        result = self._run(fresh_domain_cache)
        assert "apewisdom.io" in result.sources
        assert "stocktwits.com" in result.sources
        assert "finance.yahoo.com" in result.sources

    def test_metadata_has_key_fields(self, fresh_domain_cache: _DomainCache) -> None:
        result = self._run(fresh_domain_cache)
        assert result.metadata["ticker"] == "AAPL"
        assert "apewisdom_rank" in result.metadata
        assert "stocktwits_bullish_pct" in result.metadata
        assert "news_count" in result.metadata

    def test_result_cached(self, fresh_cache: Cache, fresh_domain_cache: _DomainCache) -> None:
        self._run(fresh_domain_cache)
        assert fresh_cache.get("agent:sentiment:AAPL") is not None

    def test_confidence_matches_score(self, fresh_domain_cache: _DomainCache) -> None:
        from aer.agents.sentiment import _score_confidence

        snap = _snap()
        expected = _score_confidence(snap)
        result = self._run(fresh_domain_cache, snap=snap)
        assert result.confidence == pytest.approx(expected, abs=0.01)

    def test_uses_analysis_tier(self, fresh_domain_cache: _DomainCache) -> None:
        mock_response = MagicMock()
        mock_response.content = "text"
        with patch("aer.agents.sentiment.agent_cache", return_value=fresh_domain_cache):
            with patch("aer.agents.sentiment.get_sentiment_snapshot", return_value=_snap()):
                with patch("aer.agents.sentiment.complete") as mock_complete:
                    mock_complete.return_value = mock_response
                    from aer.agents.sentiment import run

                    run("AAPL")
        assert mock_complete.call_args.kwargs["tier"] == "ANALYSIS"
