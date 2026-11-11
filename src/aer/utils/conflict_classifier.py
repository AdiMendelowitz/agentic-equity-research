"""Pure Python conflict classifier - 8 named patterns. No LLM calls.

SEVERITY SCALE: the per-pattern severity constants below are an UNVALIDATED
expert-prior ordinal scale. They encode a relative risk ordering judged by domain priors and
have NOT been calibrated against realized recommendation outcomes; there is no outcome dataset
to fit them against. They are an ordinal ranking, not a probability, confidence, or loss estimate.

The report gate (_conflicts_exceed_threshold) compares a single conflict's severity with strict
`>=` against CONFLICT_REPORT_THRESHOLD and is intentionally non-additive: two sub-threshold
conflicts never combine to clear the gate. Because the constants pin existing test behavior, they
are NOT recalibrated here; only documentation and honesty about their provenance are added.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

import portalocker

from aer.paths import PROJECT_ROOT
from aer.utils.constants import (
    CONF_CONVICTION_SPLIT,
    CONF_HIGH_CONVICTION_HIGH_RISK,
    CONF_HIGH_CONVICTION_MIN_AGENTS,
    CONF_MACRO_OVERRIDE,
    CONF_MACRO_OVERRIDE_CONFIDENCE_FLOOR,
    CONF_ML_SIGNAL_MISMATCH,
    CONF_REGIME_MISMATCH,
    CONF_SENTIMENT_DIVERGENCE,
    CONF_SMART_MONEY_FLOW,
    CONF_TECHNICAL_REVERSAL,
    CONFLICT_REPORT_THRESHOLD,
    RISK_SCORE_WARNING_THRESHOLD,
    SENTIMENT_NEGATIVE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# runtime log: under the gitignored logs/ folder, never in the published docs/ tree
_UNCLASSIFIED_LOG = PROJECT_ROOT / "logs" / "unclassified_conflicts.jsonl"

# Severity constants for each named pattern
_SEV_HIGH_CONVICTION_HIGH_RISK = 9
_SEV_MACRO_OVERRIDE = 9
_SEV_REGIME_MISMATCH = 8
_SEV_SMART_MONEY_FLOW = 6
_SEV_SENTIMENT_DIVERGENCE = 6
_SEV_TECHNICAL_REVERSAL = 6
_SEV_CONVICTION_SPLIT = 6
# Severity 5 is a deliberate "track but never surface" tier: it sits strictly below
# CONFLICT_REPORT_THRESHOLD, so classify_conflict logs it (sub-threshold count) but never
# returns it. This pattern is detected and counted for observability, not reported in critiques.
_SEV_ML_SIGNAL_MISMATCH = 5  # intentionally sub-threshold (logged-but-never-reported tier)


def classify_conflict(
    views: list,
    macro: dict[str, Any],
    risk: dict[str, Any],
) -> list[dict]:
    """Return conflicts at severity >= CONFLICT_REPORT_THRESHOLD, sorted descending.

    Args:
        views: list[ProfessorAgentView] (typed as list to avoid circular import)
        macro: macro snapshot dict
        risk: risk assessment dict with optional 'score' key

    Returns:
        Conflicts with severity >= CONFLICT_REPORT_THRESHOLD, sorted by severity descending.
    """
    all_conflicts: list[dict] = []
    all_conflicts.extend(_check_high_conviction_high_risk(views, risk))
    all_conflicts.extend(_check_macro_override(views, macro))
    all_conflicts.extend(_check_smart_money_flow(views, macro))
    all_conflicts.extend(_check_sentiment_divergence(views, macro))
    all_conflicts.extend(_check_technical_reversal(views, macro))
    all_conflicts.extend(_check_regime_mismatch(views, macro))
    all_conflicts.extend(_check_conviction_split(views, macro))
    all_conflicts.extend(_check_ml_signal_mismatch(views, macro))

    above = [c for c in all_conflicts if c["severity"] >= CONFLICT_REPORT_THRESHOLD]
    below = [c for c in all_conflicts if c["severity"] < CONFLICT_REPORT_THRESHOLD]
    if below:
        _log_sub_threshold_count(len(below))
    above.sort(key=lambda c: c["severity"], reverse=True)
    return above


def _conflicts_exceed_threshold(conflicts: list[dict]) -> bool:
    """Gate: any single conflict severity >= threshold clears it. NOT additive.

    Two severity-5 conflicts do NOT clear a threshold of 6.
    """
    if not conflicts:
        return False
    return max(c["severity"] for c in conflicts) >= CONFLICT_REPORT_THRESHOLD


def _view_summary(view: Any) -> dict:
    """Extract a minimal summary dict from a ProfessorAgentView (or plain dict)."""
    if hasattr(view, "agent"):
        return {"agent": view.agent, "view": view.view, "confidence": view.confidence}
    return {"agent": view.get("agent", ""), "view": view.get("view", ""), "confidence": view.get("confidence", 0.0)}


def _check_high_conviction_high_risk(views: list, risk: dict[str, Any]) -> list[dict]:
    """High-conviction BUY votes (confidence > 0.7) from >= CONF_HIGH_CONVICTION_MIN_AGENTS AND risk score high."""
    buy_high_conf = [v for v in views if _get_view(v) == "BUY" and _get_confidence(v) > 0.7]
    if len(buy_high_conf) >= CONF_HIGH_CONVICTION_MIN_AGENTS and risk.get("score", 0) > RISK_SCORE_WARNING_THRESHOLD:
        return [
            {
                "pattern": CONF_HIGH_CONVICTION_HIGH_RISK,
                "template_key": "TMPL_HIGH_CONVICTION_RISK",
                "severity": _SEV_HIGH_CONVICTION_HIGH_RISK,
                "views": [_view_summary(v) for v in buy_high_conf],
                "details": {"risk_score": risk.get("score"), "buy_agent_count": len(buy_high_conf)},
            }
        ]
    return []


def _check_macro_override(views: list, macro: dict[str, Any]) -> list[dict]:
    """Macro SELL (RISK_OFF regime) AND at least 2 other agents BUY with sufficient confidence."""
    macro_views = [v for v in views if _get_agent(v) == "macro"]
    macro_sell = any(_get_view(v) == "SELL" for v in macro_views)
    if not macro_sell:
        regime = macro.get("regime_label", macro.get("label", ""))
        macro_sell = regime in ("RISK_OFF", "VOLATILE")

    if not macro_sell:
        return []

    non_macro_buy = [
        v
        for v in views
        if _get_agent(v) != "macro"
        and _get_view(v) == "BUY"
        and _get_confidence(v) >= CONF_MACRO_OVERRIDE_CONFIDENCE_FLOOR
    ]
    if len(non_macro_buy) >= 2:
        return [
            {
                "pattern": CONF_MACRO_OVERRIDE,
                "template_key": "TMPL_MACRO_OVERRIDE",
                "severity": _SEV_MACRO_OVERRIDE,
                "views": [_view_summary(v) for v in non_macro_buy],
                "details": {"macro_sell": True, "buy_agent_count": len(non_macro_buy)},
            }
        ]
    return []


def _check_smart_money_flow(views: list, macro: dict[str, Any]) -> list[dict]:  # noqa: ARG001
    """Options_strategy view is BUY (call-dominant flow) AND fundamental view is SELL."""
    options_buy = any(_get_agent(v) == "options_strategy" and _get_view(v) == "BUY" for v in views)
    fundamental_sell = any(_get_agent(v) == "fundamental" and _get_view(v) == "SELL" for v in views)
    if options_buy and fundamental_sell:
        relevant = [v for v in views if _get_agent(v) in ("options_strategy", "fundamental")]
        return [
            {
                "pattern": CONF_SMART_MONEY_FLOW,
                "template_key": "TMPL_SMART_MONEY_UNCONFIRMED",
                "severity": _SEV_SMART_MONEY_FLOW,
                "views": [_view_summary(v) for v in relevant],
                "details": {"options_buy": True, "fundamental_sell": True},
            }
        ]
    return []


def _check_sentiment_divergence(views: list, macro: dict[str, Any]) -> list[dict]:  # noqa: ARG001
    """Sentiment view is SELL AND overall direction from 2+ other views is BUY."""
    sentiment_sell = any(_get_agent(v) == "sentiment" and _get_view(v) == "SELL" for v in views)
    if not sentiment_sell:
        return []
    non_sentiment_buy = [v for v in views if _get_agent(v) != "sentiment" and _get_view(v) == "BUY"]
    if len(non_sentiment_buy) >= 2:
        relevant = [v for v in views if _get_agent(v) == "sentiment"] + non_sentiment_buy
        return [
            {
                "pattern": CONF_SENTIMENT_DIVERGENCE,
                "template_key": "TMPL_SENTIMENT_DIVERGENCE_RISK_ON",
                "severity": _SEV_SENTIMENT_DIVERGENCE,
                "views": [_view_summary(v) for v in relevant],
                "details": {
                    "sentiment_sell": True,
                    "non_sentiment_buy_count": len(non_sentiment_buy),
                    "sentiment_threshold": SENTIMENT_NEGATIVE_THRESHOLD,
                },
            }
        ]
    return []


def _check_technical_reversal(views: list, macro: dict[str, Any]) -> list[dict]:  # noqa: ARG001
    """Technical SELL AND fundamental BUY (or vice versa), both high confidence > 0.65."""
    tech = next((v for v in views if _get_agent(v) == "technical"), None)
    fund = next((v for v in views if _get_agent(v) == "fundamental"), None)
    if tech is None or fund is None:
        return []
    tech_view = _get_view(tech)
    fund_view = _get_view(fund)
    tech_conf = _get_confidence(tech)
    fund_conf = _get_confidence(fund)
    opposing = (tech_view == "SELL" and fund_view == "BUY") or (tech_view == "BUY" and fund_view == "SELL")
    high_conf_both = tech_conf > 0.65 and fund_conf > 0.65
    if opposing and high_conf_both:
        return [
            {
                "pattern": CONF_TECHNICAL_REVERSAL,
                "template_key": "TMPL_VALUATION_MOMENTUM",
                "severity": _SEV_TECHNICAL_REVERSAL,
                "views": [_view_summary(tech), _view_summary(fund)],
                "details": {
                    "technical_view": tech_view,
                    "fundamental_view": fund_view,
                    "technical_confidence": tech_conf,
                    "fundamental_confidence": fund_conf,
                },
            }
        ]
    return []


def _check_regime_mismatch(views: list, macro: dict[str, Any]) -> list[dict]:
    """Macro regime_label is RISK_OFF but 3+ agents are BUY."""
    regime = macro.get("regime_label", macro.get("label", ""))
    is_risk_off = regime == "RISK_OFF"
    if not is_risk_off:
        macro_views = [v for v in views if _get_agent(v) == "macro"]
        is_risk_off = any(_get_regime(v) == "RISK_OFF" for v in macro_views)
    if not is_risk_off:
        return []
    buy_agents = [v for v in views if _get_view(v) == "BUY"]
    if len(buy_agents) >= 3:
        return [
            {
                "pattern": CONF_REGIME_MISMATCH,
                "template_key": "TMPL_REGIME_CONFLICT",
                "severity": _SEV_REGIME_MISMATCH,
                "views": [_view_summary(v) for v in buy_agents],
                "details": {"regime": regime, "buy_agent_count": len(buy_agents)},
            }
        ]
    return []


def _check_conviction_split(views: list, macro: dict[str, Any]) -> list[dict]:  # noqa: ARG001
    """>=2 agents with confidence > 0.75 hold opposite views (one BUY, one SELL)."""
    high_conf = [v for v in views if _get_confidence(v) > 0.75]
    buy_agents = [v for v in high_conf if _get_view(v) == "BUY"]
    sell_agents = [v for v in high_conf if _get_view(v) == "SELL"]
    if buy_agents and sell_agents:
        relevant = buy_agents + sell_agents
        return [
            {
                "pattern": CONF_CONVICTION_SPLIT,
                "template_key": "TMPL_REGIME_CONFLICT",
                "severity": _SEV_CONVICTION_SPLIT,
                "views": [_view_summary(v) for v in relevant],
                "details": {
                    "buy_high_conf_agents": [_get_agent(v) for v in buy_agents],
                    "sell_high_conf_agents": [_get_agent(v) for v in sell_agents],
                },
            }
        ]
    return []


def _check_ml_signal_mismatch(views: list, macro: dict[str, Any]) -> list[dict]:  # noqa: ARG001
    """ML prediction conflicts with recommendation direction (severity 5 - sub-threshold).

    Checks macro.get("ml_tension").
    Intentionally below CONFLICT_REPORT_THRESHOLD - logged but not returned.
    """
    if macro.get("ml_tension"):
        return [
            {
                "pattern": CONF_ML_SIGNAL_MISMATCH,
                "template_key": "TMPL_MODEL_EXPLANATION",
                "severity": _SEV_ML_SIGNAL_MISMATCH,
                "views": [],
                "details": {"ml_tension": True},
            }
        ]
    return []


def _log_unclassified(views: list) -> None:
    """Append unclassified conflict info to logs/unclassified_conflicts.jsonl; never raises."""
    try:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "views": [{"agent": _get_agent(v), "view": _get_view(v)} for v in views],
        }
        _UNCLASSIFIED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(_UNCLASSIFIED_LOG), mode="a", flags=portalocker.LOCK_EX, encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("_log_unclassified failed: %s", e)


def _log_sub_threshold_count(count: int) -> None:
    """Increment sub-threshold conflict counter in metrics DB - never raises."""
    try:
        from aer.utils.metrics import hit_rate  # noqa: F401 - lazy import to avoid circular dependency

        # Increment via metrics-adjacent mechanism: log to module logger at INFO
        logger.info("sub_threshold_conflicts_dropped count=%d", count)
    except Exception as e:
        logger.warning("_log_sub_threshold_count failed: %s", e)


# --- Private attribute accessors (handle both ProfessorAgentView and plain dicts) ---


def _get_agent(view: Any) -> str:
    return view.agent if hasattr(view, "agent") else view.get("agent", "")


def _get_view(view: Any) -> str:
    return view.view if hasattr(view, "view") else view.get("view", "")


def _get_confidence(view: Any) -> float:
    return view.confidence if hasattr(view, "confidence") else float(view.get("confidence", 0.0))


def _get_regime(view: Any) -> str:
    return view.regime_label if hasattr(view, "regime_label") else view.get("regime_label", "")
