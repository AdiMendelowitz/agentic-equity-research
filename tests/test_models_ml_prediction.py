"""Tests for model additions."""

import pytest

from aer.models import AgentOutput, BacktestResult, InvestmentBrief, MLPrediction


def test_backtest_result_has_metadata_field():
    b = BacktestResult(ticker="AAPL")
    assert isinstance(b.metadata, dict)
    assert b.metadata == {}


def test_backtest_result_metadata_stores_arbitrary_data():
    b = BacktestResult(ticker="AAPL", metadata={"signal": "BUY"})
    assert b.metadata["signal"] == "BUY"


def test_ml_prediction_json_roundtrip():
    ml = MLPrediction(
        ticker="AAPL",
        direction="UP",
        prob_up_raw=0.63,
        cv_f1_macro=0.58,
        cv_f1_std=0.08,
        model_used="lightgbm",
        trained_on_bars=500,
        n_features=36,
        class_balance_up=0.55,
        brier_score_raw=0.22,
        agrees_with_llm=True,
        prediction_skipped=False,
    )
    data = ml.model_dump(mode="json")
    restored = MLPrediction.model_validate(data)
    assert restored.ticker == "AAPL"
    assert restored.direction == "UP"
    assert restored.prob_up_raw == pytest.approx(0.63)
    assert restored.model_used == "lightgbm"


def test_ml_prediction_skipped_uses_none_model():
    ml = MLPrediction(ticker="AAPL", prediction_skipped=True, model_used="none")
    assert ml.model_used == "none"
    assert ml.prediction_skipped is True


def test_ml_prediction_is_frozen():
    ml = MLPrediction(ticker="AAPL", model_used="logistic")
    with pytest.raises(Exception):
        ml.ticker = "GOOG"  # type: ignore


def test_investment_brief_has_ml_prediction_field():
    brief = InvestmentBrief(ticker="AAPL", company_name="Apple")
    assert brief.ml_prediction is None


def test_investment_brief_accepts_ml_prediction():
    ml = MLPrediction(ticker="AAPL", direction="DOWN", prob_up_raw=0.35, model_used="logistic")
    brief = InvestmentBrief(ticker="AAPL", company_name="Apple", ml_prediction=ml)
    assert brief.ml_prediction is not None
    assert brief.ml_prediction.direction == "DOWN"


def test_get_vix_regime_from_options_strategy():
    from aer.models import OptionsStrategyOutput, RegimeOutput

    regime = RegimeOutput(
        agent_name="regime",
        content="test",
        vix_current=18.5,
        vix_regime="NORMAL",
        iv_rank_proxy=0.4,
        trend_state="TRENDING_UP",
        iv_edge="SELLER_EDGE",
        gap_risk="LOW",
        verdict="GREEN",
        verdict_rationale="ok",
        macro_event_imminent=False,
        macro_event_name="",
        strategy_recommendation="sell puts",
    )
    opts = OptionsStrategyOutput(agent_name="opts", content="test", regime=regime)
    brief = InvestmentBrief(ticker="AAPL", company_name="Apple", options_strategy=opts)
    assert brief.get_vix_regime() == "NORMAL"


def test_get_vix_regime_fallback_to_macro_metadata():
    macro = AgentOutput(agent_name="macro", content="test", metadata={"vix_regime": "ELEVATED"})
    brief = InvestmentBrief(ticker="AAPL", company_name="Apple", macro=macro)
    assert brief.get_vix_regime() == "ELEVATED"


def test_get_vix_regime_returns_none_when_absent():
    brief = InvestmentBrief(ticker="AAPL", company_name="Apple")
    assert brief.get_vix_regime() is None


# ---------------------------------------------------------------------------
# MLPrediction ensemble fields
# ---------------------------------------------------------------------------


class TestMLPredictionEnsembleFields:
    def test_new_fields_default_none(self):
        ml = MLPrediction(ticker="X")
        assert ml.calibrated_prob_up is None
        assert ml.ensemble_prob_up is None
        assert ml.ensemble_direction is None

    def test_ensemble_fields_model_copy(self):
        ml = MLPrediction(ticker="X")
        ml2 = ml.model_copy(update={"calibrated_prob_up": 0.63, "ensemble_prob_up": 0.61, "ensemble_direction": "UP"})
        assert ml2.calibrated_prob_up == pytest.approx(0.63)
        assert ml2.ensemble_prob_up == pytest.approx(0.61)
        assert ml2.ensemble_direction == "UP"
        assert ml.calibrated_prob_up is None
        assert ml.ensemble_prob_up is None
        assert ml.ensemble_direction is None

    def test_ensemble_fields_round_trip(self):
        ml = MLPrediction(
            ticker="X",
            calibrated_prob_up=0.63,
            ensemble_prob_up=0.61,
            ensemble_direction="UP",
        )
        dumped = ml.model_dump(mode="json")
        restored = MLPrediction.model_validate(dumped)
        assert restored.calibrated_prob_up == pytest.approx(0.63)
        assert restored.ensemble_prob_up == pytest.approx(0.61)
        assert restored.ensemble_direction == "UP"


class TestComputeEnsemble:
    def test_skipped_returns_empty(self):
        from aer.ml.ensemble import compute_ensemble

        lgbm = MLPrediction(ticker="X", prediction_skipped=True, model_used="none")
        logistic = MLPrediction(ticker="X", prediction_skipped=True, model_used="none")
        assert compute_ensemble(lgbm, logistic, 0.55, 0.55) == {}

    def test_equal_f1_weights(self):
        from aer.ml.ensemble import compute_ensemble

        lgbm = MLPrediction(ticker="X", prob_up_raw=0.6, model_used="lightgbm")
        logistic = MLPrediction(ticker="X", prob_up_raw=0.4, model_used="logistic")
        result = compute_ensemble(lgbm, logistic, 0.55, 0.55)
        assert result["ensemble_prob_up"] == pytest.approx(0.5, abs=1e-4)
        assert result["ensemble_direction"] == "UP"  # 0.5 >= 0.5 -> UP

    def test_calibrated_prob_used_over_raw(self):
        from aer.ml.ensemble import compute_ensemble

        lgbm = MLPrediction(ticker="X", prob_up_raw=0.4, calibrated_prob_up=0.7, model_used="lightgbm")
        logistic = MLPrediction(ticker="X", prob_up_raw=0.4, model_used="logistic")
        result = compute_ensemble(lgbm, logistic, 0.6, 0.4)
        assert result["ensemble_direction"] == "UP"

    def test_zero_f1_sum_no_error(self):
        from aer.ml.ensemble import compute_ensemble

        lgbm = MLPrediction(ticker="X", prob_up_raw=0.6, model_used="lightgbm")
        logistic = MLPrediction(ticker="X", prob_up_raw=0.4, model_used="logistic")
        result = compute_ensemble(lgbm, logistic, 0.0, 0.0)
        assert isinstance(result, dict)
        assert "ensemble_prob_up" in result

    def test_none_prob_returns_empty(self):
        from aer.ml.ensemble import compute_ensemble

        lgbm = MLPrediction(ticker="X", prob_up_raw=0.6, model_used="lightgbm")
        logistic = MLPrediction(ticker="X", prob_up_raw=None, model_used="logistic")
        assert compute_ensemble(lgbm, logistic, 0.55, 0.55) == {}

    def test_direction_down(self):
        from aer.ml.ensemble import compute_ensemble

        lgbm = MLPrediction(ticker="X", prob_up_raw=0.45, model_used="lightgbm")
        logistic = MLPrediction(ticker="X", prob_up_raw=0.45, model_used="logistic")
        result = compute_ensemble(lgbm, logistic, 0.55, 0.55)
        assert result["ensemble_direction"] == "DOWN"
