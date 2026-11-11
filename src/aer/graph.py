"""LangGraph pipeline: ticker -> parallel agents -> synthesis -> InvestmentBrief."""

import json
import logging
import threading
import warnings
from typing import Annotated, Any, TypedDict

logger = logging.getLogger(__name__)

# langgraph.checkpoint.serde.jsonplus constructs Reviver() at module level without
# allowed_objects - a library bug in LangGraph 1.x. Suppress until they fix it upstream.
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from operator import add

from langgraph.graph import END, START, StateGraph


def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts, with b's values overwriting a's."""
    result = a.copy()
    result.update(b)
    return result


from datetime import date as _date

from aer.config import get_settings
from aer.data.opinions import record_opinion
from aer.data.prices import get_company_info, get_current_price  # get_current_price kept as fallback
from aer.ml.constants import ML_NODE_TIMEOUT_SECONDS, REC_TO_DIRECTION
from aer.models import (
    AgentOutput,
    InvestmentBrief,
    MLPrediction,
    OpinionRecord,
    SynthesisOutput,
)
from aer.risk.models import FactorLoadings
from aer.utils.cache import agent_cache, get_recommendation_log_store
from aer.utils.llm_client import complete
from aer.utils.prompt_builders import (
    build_synthesis_json_instruction,
    build_synthesis_system_prompt,
    build_synthesis_user_message,
    strip_json_markdown_fences,
)
from aer.utils.vintage_tracker import clear_session, get_session_vintage, start_session

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class BriefState(TypedDict):
    ticker: str
    fundamental: AgentOutput | None
    technical: AgentOutput | None
    macro: AgentOutput | None
    sentiment: AgentOutput | None
    positioning: AgentOutput | None
    company_info: dict | None
    brief: InvestmentBrief | None
    ml_prediction: MLPrediction | None
    news: AgentOutput | None
    factor_loadings: FactorLoadings | None
    # Track failed agents for synthesis validation
    failed_agents: Annotated[list[str], add]  # Agents that raised exceptions (reducer: concat lists)
    agent_errors: Annotated[dict[str, str], _merge_dicts]  # Agent name → error message mapping
    synthesis_model_used: str | None  # Actual model identifier from LLM response


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------


def _run_fundamental(state: BriefState) -> dict:
    from aer.agents import fundamental  # local import (matches newer node pattern)

    try:
        result = fundamental.run(state["ticker"])
    except Exception as e:
        logger.warning("Fundamental agent failed for %s: %s", state["ticker"], e)
        # Track agent failure for synthesis validation
        failed = state.get("failed_agents", [])
        errors = state.get("agent_errors", {})
        return {
            "fundamental": None,
            "failed_agents": failed + ["fundamental"],
            "agent_errors": {**errors, "fundamental": str(e)},
        }
    # get_company_info is cached by fundamental.run(); read into state so _synthesize
    # doesn't re-fetch. Isolated try so a failure here doesn't discard result.
    try:
        info = get_company_info(state["ticker"])
        return {"fundamental": result, "company_info": info}
    except Exception as e:
        logger.warning("Company info fetch failed for %s: %s", state["ticker"], e)
        return {"fundamental": result}


def _run_technical(state: BriefState) -> dict:
    from aer.agents import technical  # local import (matches newer node pattern)

    try:
        return {"technical": technical.run(state["ticker"])}
    except Exception as e:
        logger.warning("Technical agent failed for %s: %s", state["ticker"], e)
        # Track agent failure for synthesis validation
        failed = state.get("failed_agents", [])
        errors = state.get("agent_errors", {})
        return {
            "technical": None,
            "failed_agents": failed + ["technical"],
            "agent_errors": {**errors, "technical": str(e)},
        }


def _run_macro(state: BriefState) -> dict:
    from aer.agents import macro  # local import (matches newer node pattern)

    try:
        return {"macro": macro.run(state["ticker"])}
    except Exception as e:
        logger.warning("Macro agent failed for %s: %s", state["ticker"], e)
        # Track agent failure for synthesis validation
        failed = state.get("failed_agents", [])
        errors = state.get("agent_errors", {})
        return {
            "macro": None,
            "failed_agents": failed + ["macro"],
            "agent_errors": {**errors, "macro": str(e)},
        }


def _run_sentiment(state: BriefState) -> dict:
    from aer.agents import sentiment  # local import (matches newer node pattern)

    try:
        return {"sentiment": sentiment.run(state["ticker"])}
    except Exception as e:
        logger.warning("Sentiment agent failed for %s: %s", state["ticker"], e)
        # Track agent failure for synthesis validation
        failed = state.get("failed_agents", [])
        errors = state.get("agent_errors", {})
        return {
            "sentiment": None,
            "failed_agents": failed + ["sentiment"],
            "agent_errors": {**errors, "sentiment": str(e)},
        }


def _run_positioning(state: BriefState) -> dict:
    from aer.agents import positioning  # local import (matches newer node pattern)

    try:
        return {"positioning": positioning.run(state["ticker"])}
    except Exception as e:
        logger.warning("Positioning agent failed for %s: %s", state["ticker"], e)
        # Track agent failure for synthesis validation
        failed = state.get("failed_agents", [])
        errors = state.get("agent_errors", {})
        return {
            "positioning": None,
            "failed_agents": failed + ["positioning"],
            "agent_errors": {**errors, "positioning": str(e)},
        }


def _run_news(state: BriefState) -> dict:
    from aer.agents import news  # local import (matches newer node pattern)

    try:
        return {"news": news.run(state["ticker"])}
    except Exception as e:
        logger.warning("News agent failed for %s: %s", state["ticker"], e)
        return {"news": None}


def _agent_signal_to_opinion_signal(
    agent_output: AgentOutput,
    agent_name: str,
    brief: InvestmentBrief,
) -> str:
    """Derive opinion signal from agent output. Falls back to brief recommendation."""
    # For most agents, use brief.recommendation (LLM synthesis)
    # Exception: insider_buying has its own signal logic
    if agent_name == "insider_buying" and hasattr(agent_output, "signal_label"):
        # insider_buying uses signal_label: STRONG_BUY → BUY, BUY → BUY, WATCH → HOLD, NOISE → AVOID
        signal_map = {"STRONG_BUY": "BUY", "BUY": "BUY", "WATCH": "HOLD", "NOISE": "AVOID"}
        return signal_map.get(agent_output.signal_label, "HOLD")
    # Default: use brief recommendation (LLM synthesis)
    return brief.recommendation


def _post_process(state: BriefState) -> dict:
    """Enrich the brief with price targets, factor loadings and walk-forward results.

    All post-synthesis mutations happen here inside the graph rather than in
    the caller - keeps every state transition inside the state machine.
    """
    brief = state.get("brief")
    if brief is None:
        return {}

    try:
        from aer.data.price_targets import compute_price_targets  # local import - pure compute, no side effects

        targets = compute_price_targets(brief.ticker)
        if targets:
            brief.entry_price = targets.get("entry_price")
            brief.entry_method = targets.get("entry_method")
            brief.target_price = targets.get("target_price")
            brief.target_method = targets.get("target_method")
            brief.stop_loss_price = targets.get("stop_loss_price")
            brief.risk_reward_ratio = targets.get("risk_reward_ratio")
    except Exception as e:
        logger.warning("Price target computation failed for %s: %s", brief.ticker, e)

    try:
        from aer.risk.factors import get_factor_loadings  # local import - statsmodels is optional

        brief.factor_loadings = get_factor_loadings(brief.ticker, as_of_date=brief.date.date())
    except Exception as e:
        logger.warning("Factor loadings failed for %s: %s", brief.ticker, e)

    try:
        from aer.risk.walkforward import run_walk_forward  # local import

        brief.walk_forward = run_walk_forward(brief.ticker, as_of_date=brief.date.date())
    except Exception as e:
        logger.warning("Walk-forward validation failed for %s: %s", brief.ticker, e)

    # Record opinions for all 8 agents
    agents_to_record = [
        ("fundamental", state.get("fundamental")),
        ("technical", state.get("technical")),
        ("sentiment", state.get("sentiment")),
        ("macro", state.get("macro")),
        ("positioning", state.get("positioning")),
        ("insider_buying", state.get("insider_buying")),
        ("pelosi", state.get("pelosi")),
    ]

    for agent_name, agent_output in agents_to_record:
        if agent_output is None:
            # Record neutral opinion for absent agents (safety)
            opinion = OpinionRecord(
                ticker=brief.ticker,
                agent_name=agent_name,
                opinion_date=brief.date.date(),
                signal="HOLD",
                confidence=0.0,
                conviction=0.0,
                drivers=[],
                forward_return_5d=None,
            )
        else:
            # Extract signal from agent output
            signal = _agent_signal_to_opinion_signal(agent_output, agent_name, brief)
            drivers = agent_output.sources[:3] if hasattr(agent_output, "sources") and agent_output.sources else []
            confidence = (
                agent_output.confidence
                if hasattr(agent_output, "confidence") and agent_output.confidence is not None
                else 0.5
            )

            opinion = OpinionRecord(
                ticker=brief.ticker,
                agent_name=agent_name,
                opinion_date=brief.date.date(),
                signal=signal,
                confidence=confidence,
                conviction=brief.conviction if brief.conviction is not None else 0.5,
                drivers=drivers,
                forward_return_5d=None,
            )

        # Record and check return value
        success = record_opinion(opinion)
        if not success:
            logger.warning("Failed to record opinion for %s agent %s", brief.ticker, agent_name)

    return {"brief": brief}


def _run_ml_prediction(state: BriefState) -> dict:
    """Run ML direction classifier after _post_process so brief is fully enriched.

    Runs after _post_process (not after backtesting).
    executor.shutdown(wait=False, cancel_futures=True): a timeout does not block the pipeline.
    Checks the MLPrediction cache before retraining (TTL_ANALYSIS = 43200s).
    """
    import concurrent.futures

    from aer.ml.ensemble import compute_ensemble
    from aer.ml.features import build_feature_matrix
    from aer.ml.models import train_and_predict

    brief = state.get("brief")
    if brief is None:
        return {}

    ticker = brief.ticker
    cache = agent_cache()
    today_str = _date.today().isoformat()
    cache_key = f"ml:{ticker}:{today_str}"

    cached = cache.get(cache_key)
    if cached is not None:
        try:
            ml = MLPrediction.model_validate(cached)
            if ml.direction is not None and not ml.prediction_skipped:
                llm_dir = REC_TO_DIRECTION.get(brief.recommendation)
                agrees = (llm_dir == ml.direction) if llm_dir is not None else None
                bs = brief.backtesting.metadata.get("signal") if brief.backtesting else None
                sharpe = brief.backtesting.sharpe if brief.backtesting else None
                tension: float | None = (
                    abs(ml.prob_up_raw - 0.5) if brief.recommendation == "HOLD" and ml.prob_up_raw is not None else None
                )
                ml = ml.model_copy(
                    update={
                        "agrees_with_llm": agrees,
                        "backtest_signal": bs,
                        "backtest_sharpe": sharpe,
                        "ml_tension": tension,
                    }
                )
            brief.ml_prediction = ml
            return {"brief": brief}
        except Exception as e:
            logger.warning("Cached MLPrediction invalid for %s: %s — retraining", ticker, e)

    ml: MLPrediction | None = None
    lgbm_pred: MLPrediction | None = None
    logistic_pred: MLPrediction | None = None

    def _compute() -> "tuple[MLPrediction, MLPrediction] | None":
        return train_and_predict(build_feature_matrix(ticker), ticker)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_compute)
    try:
        result = future.result(timeout=ML_NODE_TIMEOUT_SECONDS)
        if result is not None:
            lgbm_pred, logistic_pred = result
            ml = lgbm_pred
    except concurrent.futures.TimeoutError:
        logger.warning("ML prediction timeout for %s", ticker)
    except Exception as exc:
        logger.warning("ML prediction failed for %s: %s", ticker, exc)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)  # don't block pipeline

    if ml is not None and ml.direction is not None and not ml.prediction_skipped:
        llm_dir = REC_TO_DIRECTION.get(brief.recommendation)
        agrees = (llm_dir == ml.direction) if llm_dir is not None else None
        bs = brief.backtesting.metadata.get("signal") if brief.backtesting else None
        sharpe = brief.backtesting.sharpe if brief.backtesting else None
        tension: float | None = (
            abs(ml.prob_up_raw - 0.5) if brief.recommendation == "HOLD" and ml.prob_up_raw is not None else None
        )
        ml = ml.model_copy(
            update={
                "agrees_with_llm": agrees,
                "backtest_signal": bs,
                "backtest_sharpe": sharpe,
                "ml_tension": tension,
            }
        )
        # Compute ensemble if both per-model predictions are available
        if lgbm_pred is not None and logistic_pred is not None:
            f1_lgbm = lgbm_pred.cv_f1_macro if lgbm_pred.cv_f1_macro is not None else 0.0
            f1_logistic = logistic_pred.cv_f1_macro if logistic_pred.cv_f1_macro is not None else 0.0
            ensemble_fields = compute_ensemble(lgbm_pred, logistic_pred, f1_lgbm, f1_logistic)
            if ensemble_fields:
                ml = ml.model_copy(update=ensemble_fields)
            else:
                logger.warning(
                    "Ensemble skipped for %s — lgbm_skipped=%s logistic_skipped=%s",
                    ticker,
                    lgbm_pred.prediction_skipped,
                    logistic_pred.prediction_skipped,
                )

    if ml is not None:
        try:
            cache.set(cache_key, ml.model_dump(mode="json"))
        except Exception as e:
            logger.warning("Failed to cache MLPrediction for %s: %s", ticker, e)

    brief.ml_prediction = ml
    return {"brief": brief}


# ---------------------------------------------------------------------------
# Synthesis node
# ---------------------------------------------------------------------------


def _synthesize(state: BriefState) -> dict:
    ticker = state["ticker"]

    # Check for failed agents before synthesis
    failed = state.get("failed_agents", [])
    if failed:
        logger.warning("Synthesis proceeding with %d failed agents: %s", len(failed), failed)
    if len(failed) >= 4:  # More than 1 agent failed out of 5
        errors_msg = "; ".join(
            f"{agent}: {state.get('agent_errors', {}).get(agent, 'unknown error')}" for agent in failed
        )
        raise RuntimeError(
            f"Too many agent failures ({len(failed)}/5) for {ticker}. "
            f"Cannot synthesize with high failure rate. Errors: {errors_msg}"
        )

    agent_outputs = {key: state.get(key) for key in ("fundamental", "technical", "macro", "sentiment", "positioning")}
    populated = [k for k, v in agent_outputs.items() if v is not None]
    if not populated:
        raise RuntimeError(f"All agents failed for {ticker} — cannot synthesize without data")
    if len(populated) < 5:
        logger.warning("Only %d/5 agents available for %s: %s", len(populated), ticker, populated)
    _news = state.get("news")
    if _news is not None and _news.confidence <= 0:
        logger.info(
            "News signal suppressed for %s: confidence=0.0 (headline_count=%s)",
            ticker,
            _news.metadata.get("headline_count", "?"),
        )
        _news = None
    prompt = build_synthesis_user_message(ticker, agent_outputs, news=_news)

    system_prompt = build_synthesis_system_prompt(include_options=False)

    try:
        # LLM required: multi-source synthesis into InvestmentBrief - qualitative judgment from 5+ agent outputs
        synthesis: SynthesisOutput | None = None
        for _attempt in range(2):
            try:
                response = complete(
                    system_prompt=system_prompt,
                    user_message=prompt + build_synthesis_json_instruction(),
                    tier="SYNTHESIS",
                    max_tokens=2500 if _attempt == 1 else 2000,
                    json_mode=True,
                    temperature=1.0,  # non-deterministic by design
                )
            except ValueError as e:
                if "No text blocks" in str(e) and _attempt == 0:
                    logger.warning(
                        "Synthesis got ToolUseBlock for %s (attempt %d) — retrying with explicit no-tool instruction",
                        ticker,
                        _attempt + 1,
                    )
                    continue
                raise

            # Handle empty response
            if not response.content or not response.content.strip():
                if _attempt == 0:
                    logger.warning("Synthesis returned empty response for %s — retrying", ticker)
                    continue
                raise ValueError(f"Empty synthesis response for {ticker} after retry")

            raw = strip_json_markdown_fences(response.content.strip())
            if not raw:
                if _attempt == 0:
                    logger.warning("Synthesis returned blank JSON for %s — retrying", ticker)
                    continue
                raise ValueError(f"Synthesis returned blank JSON for {ticker} after retry")
            try:
                data, _ = json.JSONDecoder().raw_decode(raw)  # raw_decode ignores trailing text after the JSON object
            except json.JSONDecodeError as je:
                if _attempt == 0:
                    logger.warning("Synthesis returned invalid JSON for %s: %s — retrying", ticker, je)
                    continue
                raise
            _required = {"recommendation", "conviction", "summary"}
            _missing = _required - data.keys() if isinstance(data, dict) else _required
            if _missing:
                if _attempt == 0:
                    logger.warning("Synthesis missing required keys %s for %s — retrying", _missing, ticker)
                    continue
                raise ValueError(f"Synthesis JSON missing required keys after retry: {_missing}")
            # Log full JSON data structure on validation error for debugging
            try:
                synthesis = SynthesisOutput.model_validate(data)
            except ValueError:
                logger.error(
                    "SynthesisOutput validation failed for %s. JSON data was:\n%s",
                    ticker,
                    json.dumps(data, indent=2, default=str),
                )
                raise
            if not synthesis.summary:
                logger.warning("Synthesis produced empty summary for %s — possible token truncation", ticker)
            if not synthesis.catalysts or not synthesis.risks:
                if _attempt == 0:
                    logger.warning(
                        "Synthesis catalysts/risks empty for %s — possible token truncation; retrying with max_tokens=2500",
                        ticker,
                    )
                    synthesis = None
                    continue
                logger.warning("Synthesis catalysts/risks still empty for %s after retry — proceeding", ticker)
            break
        if synthesis is None:
            raise ValueError("Synthesis loop exited without producing output")
    except Exception as e:
        raise RuntimeError(f"Synthesis LLM call failed for {ticker}: {e}") from e

    fund: AgentOutput | None = state.get("fundamental")  # type: ignore[assignment]
    fund_meta = fund.metadata if fund is not None else {}
    info = state.get("company_info") or get_company_info(ticker)
    price = fund_meta.get("price") if fund_meta.get("price") is not None else get_current_price(ticker)

    def _ao(v: Any) -> "dict | None":
        # Serialize to dict so Pydantic reconstructs AgentOutput using its own class reference.
        # Guards against sys.path double-import producing two distinct AgentOutput class objects.
        return v.model_dump(mode="json") if v is not None else None

    brief = InvestmentBrief(
        ticker=ticker,
        company_name=info.get("name", ticker),
        exchange=info.get("exchange", ""),
        sector=info.get("sector", ""),
        price_currency=info.get("currency", "USD"),
        current_price=price,
        price_target=synthesis.price_target,
        recommendation=synthesis.recommendation,
        conviction=synthesis.conviction,
        summary=synthesis.summary,
        fundamental=_ao(state.get("fundamental")),
        technical=_ao(state.get("technical")),
        macro=_ao(state.get("macro")),
        sentiment=_ao(state.get("sentiment")),
        positioning=_ao(state.get("positioning")),
        news=_ao(state.get("news")),
        risks=synthesis.risks,
        catalysts=synthesis.catalysts,
        time_horizon=synthesis.time_horizon,
    )

    # Persist recommendation to the audit log for calibration tracking
    try:
        get_recommendation_log_store().log(
            ticker=ticker,
            recommendation=synthesis.recommendation,
            conviction=synthesis.conviction,
            price=price,
        )
    except Exception as e:
        logger.warning("Recommendation logging failed for %s: %s", ticker, e)

    return {"brief": brief, "synthesis_model_used": response.model_used}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    builder = StateGraph(BriefState)

    for name, fn in [
        ("fundamental", _run_fundamental),
        ("technical", _run_technical),
        ("macro", _run_macro),
        ("sentiment", _run_sentiment),
        ("positioning", _run_positioning),
    ]:
        builder.add_node(name, fn)
        builder.add_edge(START, name)
        builder.add_edge(name, "synthesize")

    # Optional news node - controlled by NEWS_AGENT_ENABLED env flag
    if get_settings().news_agent_enabled:
        builder.add_node("news", _run_news)
        builder.add_edge(START, "news")
        builder.add_edge("news", "synthesize")

    builder.add_node("synthesize", _synthesize)
    builder.add_node("post_process", _post_process)
    builder.add_edge("synthesize", "post_process")
    builder.add_node("ml_prediction", _run_ml_prediction)
    builder.add_edge("post_process", "ml_prediction")
    builder.add_edge("ml_prediction", END)

    return builder.compile()


_graph: Any = None
_graph_error: Exception | None = None
_graph_lock = threading.Lock()


def reset_graph_cache() -> None:
    """Clear the compiled graph singleton. Required after settings changes in tests or runtime flag toggles."""
    global _graph, _graph_error
    with _graph_lock:
        _graph = None
        _graph_error = None


def _get_graph() -> Any:
    global _graph, _graph_error
    if _graph_error is not None:
        raise RuntimeError(f"Graph compilation failed previously: {_graph_error}") from _graph_error
    if _graph is None:
        with _graph_lock:
            if _graph_error is not None:
                raise RuntimeError(f"Graph compilation failed previously: {_graph_error}") from _graph_error
            if _graph is None:
                try:
                    _graph = _build_graph()
                except (ImportError, SyntaxError) as e:
                    # Non-recoverable: missing dep or bad code - persist error to avoid retries.
                    _graph_error = e
                    raise RuntimeError(f"Graph compilation failed (non-recoverable): {e}") from e
                except Exception as e:
                    # Transient failure - log but do not set _graph_error so next call retries.
                    logger.warning("Graph compilation failed (transient): %s", e)
                    raise RuntimeError(f"Graph compilation failed: {e}") from e
    return _graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def invoke_pipeline(ticker: str) -> tuple[InvestmentBrief, dict]:
    """Run the graph once for `ticker`; return (brief, meta). Additive core used by run_pipeline.

    The invoke/deepcopy/vintage-merge core WITHOUT caching or calibration logging (run_pipeline
    owns persistence and the calibration log). Never mutates the returned LangGraph state.
    """
    from copy import deepcopy

    ticker = ticker.upper()
    start_session()
    try:
        initial: BriefState = {
            "ticker": ticker,
            "fundamental": None,
            "technical": None,
            "macro": None,
            "sentiment": None,
            "positioning": None,
            "company_info": None,
            "brief": None,
            "ml_prediction": None,
            "news": None,
            "factor_loadings": None,
            "failed_agents": [],
            "agent_errors": {},
        }
        final = _get_graph().invoke(initial)
        brief_from_state: InvestmentBrief | None = final.get("brief")
        if brief_from_state is None:
            raise RuntimeError(f"Pipeline produced no brief for {ticker}. Check agent logs.")
        brief = deepcopy(brief_from_state)
        brief.vintage.update(get_session_vintage())
        return brief, {
            "synthesis_model_used": final.get("synthesis_model_used"),
        }
    finally:
        clear_session()
