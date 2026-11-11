"""Tests for LangGraph pipeline: build_brief caching, SynthesisOutput validation."""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from aer.data.cache import TTL_BRIEF, Cache
from aer.models import AgentOutput, InvestmentBrief, MLPrediction
from aer.utils.cache import _DomainCache, market_aware_ttl


@pytest.fixture
def fresh_cache(tmp_path) -> Cache:
    return Cache(db_path=str(tmp_path / "graph.db"))


@pytest.fixture
def fresh_brief_cache(fresh_cache: Cache) -> _DomainCache:
    """_DomainCache wrapping fresh_cache with brief domain (callable TTL)."""
    return _DomainCache(fresh_cache, lambda: market_aware_ttl(TTL_BRIEF), "brief")


def _agent(name: str) -> AgentOutput:
    return AgentOutput(agent_name=name, content=f"{name} analysis", confidence=0.8)


def _brief(ticker: str = "AAPL") -> InvestmentBrief:
    return InvestmentBrief(
        ticker=ticker,
        company_name="Apple Inc.",
        exchange="NASDAQ",
        sector="Technology",
        current_price=185.5,
        price_target=210.0,
        recommendation="BUY",
        conviction=0.75,
        summary="Strong fundamentals and momentum.",
        fundamental=_agent("fundamental"),
        technical=_agent("technical"),
        macro=_agent("macro"),
        sentiment=_agent("sentiment"),
        positioning=_agent("positioning"),
        risks=["Rate sensitivity", "Competition"],
        catalysts=["Services growth", "AI integration"],
        time_horizon="medium",
    )


class TestSynthesisOutput:
    @pytest.mark.parametrize("rec", ["BUY", "HOLD", "SELL", "AVOID"])
    def test_valid_recommendation(self, rec: str) -> None:
        from aer.graph import SynthesisOutput

        s = SynthesisOutput(recommendation=rec, conviction=0.7, summary="test", risks=["r"], catalysts=["c"])
        assert s.recommendation == rec

    def test_invalid_recommendation_raises(self) -> None:
        from aer.graph import SynthesisOutput

        with pytest.raises(ValidationError):
            SynthesisOutput(recommendation="STRONG_BUY", conviction=0.7, summary="", risks=[], catalysts=[])

    def test_conviction_below_zero_raises(self) -> None:
        from aer.graph import SynthesisOutput

        with pytest.raises(ValidationError):
            SynthesisOutput(recommendation="HOLD", conviction=-0.1, summary="", risks=[], catalysts=[])

    def test_conviction_above_one_raises(self) -> None:
        from aer.graph import SynthesisOutput

        with pytest.raises(ValidationError):
            SynthesisOutput(recommendation="HOLD", conviction=1.1, summary="", risks=[], catalysts=[])

    def test_price_target_can_be_none(self) -> None:
        from aer.graph import SynthesisOutput

        s = SynthesisOutput(
            recommendation="HOLD", conviction=0.5, price_target=None, summary="neutral", risks=[], catalysts=[]
        )
        assert s.price_target is None

    @pytest.mark.parametrize("th", ["short", "medium", "long"])
    def test_valid_time_horizons(self, th: str) -> None:
        from aer.graph import SynthesisOutput

        s = SynthesisOutput(
            recommendation="HOLD", conviction=0.5, time_horizon=th, summary="neutral", risks=[], catalysts=[]
        )
        assert s.time_horizon == th


class TestSynthesisOptimizations:
    def test_synth_system_prompt_exceeds_1024_tokens(self) -> None:
        from aer.utils.prompt_builders import _SYNTH_SYSTEM_BASE

        token_estimate = len(_SYNTH_SYSTEM_BASE) // 4
        assert (
            token_estimate > 1024
        ), f"_SYNTH_SYSTEM_BASE is only ~{token_estimate} tokens — must exceed 1024 to qualify for ephemeral caching"

    def test_synthesize_uses_build_synthesis_user_message(self) -> None:
        import inspect

        import aer.graph as m

        source = inspect.getsource(m._synthesize)
        assert (
            "build_synthesis_user_message" in source
        ), "_synthesize must call build_synthesis_user_message to use the prompt builder"

    def test_synthesize_uses_complete_function(self) -> None:
        import inspect

        import aer.graph as m

        source = inspect.getsource(m._synthesize)
        assert "complete" in source, "_synthesize must call complete() from the unified LLM client"


def _synthesis_state() -> dict:
    """Minimal BriefState for direct _synthesize() unit testing."""
    fund = AgentOutput(
        agent_name="fundamental",
        content="test",
        confidence=0.8,
        metadata={"price": 185.5},
    )
    return {
        "ticker": "AAPL",
        "fundamental": fund,
        "technical": _agent("technical"),
        "macro": _agent("macro"),
        "sentiment": _agent("sentiment"),
        "positioning": _agent("positioning"),
        "company_info": {"name": "Apple Inc.", "exchange": "NASDAQ", "sector": "Technology", "currency": "USD"},
        "brief": None,
        "backtesting": None,
        "options_strategy": None,
    }


class TestSynthesisTruncationResilience:
    def _response(self, data: dict) -> MagicMock:
        m = MagicMock()
        m.content = json.dumps(data)
        return m

    def _full(self) -> dict:
        return {
            "recommendation": "BUY",
            "conviction": 0.7,
            "summary": "strong outlook",
            "risks": ["rate risk", "competition"],
            "catalysts": ["AI growth", "services expansion"],
        }

    def test_empty_catalysts_retries_with_higher_token_limit(self) -> None:
        """Empty catalysts on attempt 0 triggers a retry with max_tokens=2500."""
        from aer.graph import _synthesize

        responses = [
            self._response({**self._full(), "catalysts": []}),
            self._response(self._full()),
        ]
        call_kwargs: list[dict] = []

        def _complete(**kwargs):
            call_kwargs.append(kwargs)
            return responses[len(call_kwargs) - 1]

        with (
            patch("aer.graph.complete", side_effect=_complete),
            patch("aer.graph.get_recommendation_log_store") as mock_log_store,
        ):
            mock_log_store.return_value.log = MagicMock()
            _synthesize(_synthesis_state())

        assert len(call_kwargs) == 2, "expected retry — 2 complete() calls"
        assert call_kwargs[0]["max_tokens"] == 2000
        assert call_kwargs[1]["max_tokens"] == 2500

    def test_empty_risks_retries_with_higher_token_limit(self) -> None:
        """Empty risks on attempt 0 triggers a retry with max_tokens=2500."""
        from aer.graph import _synthesize

        responses = [
            self._response({**self._full(), "risks": []}),
            self._response(self._full()),
        ]
        call_kwargs: list[dict] = []

        def _complete(**kwargs):
            call_kwargs.append(kwargs)
            return responses[len(call_kwargs) - 1]

        with (
            patch("aer.graph.complete", side_effect=_complete),
            patch("aer.graph.get_recommendation_log_store") as mock_log_store,
        ):
            mock_log_store.return_value.log = MagicMock()
            _synthesize(_synthesis_state())

        assert len(call_kwargs) == 2
        assert call_kwargs[1]["max_tokens"] == 2500

    def test_full_synthesis_does_not_retry(self) -> None:
        """No retry when catalysts and risks are both populated."""
        from aer.graph import _synthesize

        call_kwargs: list[dict] = []

        def _complete(**kwargs):
            call_kwargs.append(kwargs)
            return self._response(self._full())

        with (
            patch("aer.graph.complete", side_effect=_complete),
            patch("aer.graph.get_recommendation_log_store") as mock_log_store,
        ):
            mock_log_store.return_value.log = MagicMock()
            _synthesize(_synthesis_state())

        assert len(call_kwargs) == 1, "expected no retry for complete synthesis"

    def test_warning_logged_on_empty_catalysts(self, caplog) -> None:
        """A logger.warning must be emitted when catalysts/risks are empty."""
        import logging

        from aer.graph import _synthesize

        responses = [
            self._response({**self._full(), "catalysts": []}),
            self._response(self._full()),
        ]
        idx = [0]

        def _complete(**kwargs):
            r = responses[idx[0]]
            idx[0] += 1
            return r

        with (
            patch("aer.graph.complete", side_effect=_complete),
            patch("aer.graph.get_recommendation_log_store") as mock_log_store,
        ):
            mock_log_store.return_value.log = MagicMock()
            with caplog.at_level(logging.WARNING, logger="aer.graph"):
                _synthesize(_synthesis_state())

        assert any("catalysts/risks empty" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _run_ml_prediction ensemble integration tests
# ---------------------------------------------------------------------------


def _make_ml_state(ticker: str = "AAPL", recommendation: str = "BUY") -> dict:
    """Minimal BriefState for _run_ml_prediction unit tests."""
    from aer.models import InvestmentBrief

    brief = InvestmentBrief(
        ticker=ticker,
        company_name="Apple Inc.",
        exchange="NASDAQ",
        sector="Technology",
        current_price=185.5,
        recommendation=recommendation,
        conviction=0.75,
        summary="test",
        risks=[],
        catalysts=[],
        time_horizon="medium",
    )
    return {
        "ticker": ticker,
        "fundamental": None,
        "technical": None,
        "macro": None,
        "sentiment": None,
        "positioning": None,
        "company_info": None,
        "brief": brief,
        "backtesting": None,
        "options_strategy": None,
        "ml_prediction": None,
    }


class TestRunMLPredictionEnsemble:
    """Tests that _run_ml_prediction calls compute_ensemble and stores ensemble fields."""

    def _lgbm_pred(self) -> MLPrediction:
        return MLPrediction(
            ticker="AAPL",
            direction="UP",
            prob_up_raw=0.65,
            calibrated_prob_up=0.63,
            cv_f1_macro=0.58,
            model_used="lightgbm",
            trained_on_bars=400,
        )

    def _logistic_pred(self) -> MLPrediction:
        return MLPrediction(
            ticker="AAPL",
            direction="UP",
            prob_up_raw=0.55,
            cv_f1_macro=0.54,
            model_used="logistic",
            trained_on_bars=400,
        )

    def _skipped_pred(self) -> MLPrediction:
        return MLPrediction(
            ticker="AAPL",
            prediction_skipped=True,
            model_used="lightgbm",
            trained_on_bars=0,
        )

    def test_run_ml_prediction_ensemble_fields(self, tmp_path) -> None:
        """Valid (lgbm_pred, logistic_pred) tuple → ensemble_prob_up and ensemble_direction populated."""
        from aer.data.cache import Cache
        from aer.graph import _run_ml_prediction

        lgbm_pred = self._lgbm_pred()
        logistic_pred = self._logistic_pred()
        state = _make_ml_state()

        cache = Cache(db_path=str(tmp_path / "ml_test.db"))

        with patch("aer.graph._run_ml_prediction.__wrapped__", None, create=True):
            with patch("aer.ml.models.train_and_predict", return_value=(lgbm_pred, logistic_pred)):
                with patch("aer.ml.features.build_feature_matrix", return_value=MagicMock()):
                    with patch("aer.graph.agent_cache", return_value=cache):
                        result = _run_ml_prediction(state)

        ml = result["brief"].ml_prediction
        assert ml is not None, "ml_prediction must not be None"
        assert ml.ensemble_prob_up is not None, "ensemble_prob_up must be populated"
        assert ml.ensemble_direction in ("UP", "DOWN"), "ensemble_direction must be UP or DOWN"

    def test_run_ml_prediction_skipped_returns_no_ensemble(self, tmp_path) -> None:
        """Both skipped → ensemble_prob_up stays None (compute_ensemble returns {})."""
        from aer.data.cache import Cache
        from aer.graph import _run_ml_prediction

        skipped = self._skipped_pred()
        state = _make_ml_state()
        cache = Cache(db_path=str(tmp_path / "ml_test.db"))

        with patch("aer.ml.models.train_and_predict", return_value=(skipped, skipped)):
            with patch("aer.ml.features.build_feature_matrix", return_value=MagicMock()):
                with patch("aer.graph.agent_cache", return_value=cache):
                    result = _run_ml_prediction(state)

        ml = result["brief"].ml_prediction
        # ml may be the skipped pred (not None), but ensemble_prob_up should be None
        if ml is not None:
            assert ml.ensemble_prob_up is None, "skipped predictions must not produce ensemble_prob_up"

    def test_run_ml_prediction_none_result_no_crash(self, tmp_path) -> None:
        """train_and_predict returns None → brief.ml_prediction is None, no exception."""
        from aer.data.cache import Cache
        from aer.graph import _run_ml_prediction

        state = _make_ml_state()
        cache = Cache(db_path=str(tmp_path / "ml_test.db"))

        with patch("aer.ml.models.train_and_predict", return_value=None):
            with patch("aer.ml.features.build_feature_matrix", return_value=MagicMock()):
                with patch("aer.graph.agent_cache", return_value=cache):
                    result = _run_ml_prediction(state)

        assert result["brief"].ml_prediction is None, "ml_prediction must be None when train_and_predict returns None"


# ---------------------------------------------------------------------------
# professor_critique node in graph.py
# ---------------------------------------------------------------------------


class TestProfessorCritiqueNode:
    @pytest.mark.skip(reason="component not in this slice")
    def test_professor_critique_node_registered(self) -> None:
        """_build_graph() must register a node named 'professor_critique' in the compiled graph."""
        from aer.graph import _build_graph

        graph = _build_graph()
        node_names = list(graph.nodes.keys())
        assert "professor_critique" in node_names, (
            f"'professor_critique' node missing from compiled graph. " f"Registered nodes: {node_names}"
        )

    @pytest.mark.skip(reason="component not in this slice")
    def test_run_professor_critique_returns_none_when_no_ticker(self) -> None:
        """_run_professor_critique({}) must return {'professor_critique': None} when 'brief' is absent.

        The function must not raise - it has a defensive None path for missing brief.
        """
        from unittest.mock import patch

        from aer.graph import _run_professor_critique

        empty_state: dict = {}

        # Patch the internal critique_recommendation import so no real LLM call is made.
        with patch("aer.agents.professor.critique_recommendation", return_value="mocked critique"):
            result = _run_professor_critique(empty_state)  # type: ignore[arg-type]

        assert result == {"professor_critique": None}, f"Expected {{'professor_critique': None}}, got {result!r}"
