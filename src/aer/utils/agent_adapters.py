"""ProfessorAgentView schema + 8 adapter functions + adapt_all().

Normalises heterogeneous AgentOutput metadata into a uniform ProfessorAgentView
for the conflict classifier. Called internally by professor.py — never by callers.
"""

import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from aer.models import AgentOutput

logger = logging.getLogger(__name__)


class ProfessorAgentView(BaseModel):
    model_config = {"frozen": True}

    agent: str
    ticker: str
    view: Literal["BUY", "SELL", "NEUTRAL", "ABSTAIN"]
    confidence: float
    rationale: dict
    risks: list[str]
    regime_label: str
    ts: str


def _ts(output: AgentOutput) -> str:
    return output.timestamp.isoformat() if output.timestamp else datetime.now(UTC).isoformat()


def _adapt_fundamental(output: AgentOutput) -> ProfessorAgentView:
    text = output.content.upper()
    buy_keywords = {"BUY", "STRONG_BUY", "BULLISH"}
    sell_keywords = {"SELL", "STRONG_SELL", "BEARISH"}
    has_buy = bool(re.search(r"\b(" + "|".join(buy_keywords) + r")\b", text))
    has_sell = bool(re.search(r"\b(" + "|".join(sell_keywords) + r")\b", text))
    if has_buy and not has_sell:
        view: Literal["BUY", "SELL", "NEUTRAL", "ABSTAIN"] = "BUY"
    elif has_sell and not has_buy:
        view = "SELL"
    elif has_buy and has_sell:
        view = "ABSTAIN"
    else:
        view = "ABSTAIN"
    return ProfessorAgentView(
        agent="fundamental",
        ticker=output.metadata.get("ticker", ""),
        view=view,
        confidence=output.confidence,
        rationale={"valuation": output.metadata.get("valuation_label", output.metadata.get("pe_ratio", ""))},
        risks=[],
        regime_label="",
        ts=_ts(output),
    )


def _adapt_technical(output: AgentOutput) -> ProfessorAgentView:
    overall = output.metadata.get("signals", {}).get("overall", "NEUTRAL")
    _MAP: dict[str, Literal["BUY", "SELL", "NEUTRAL", "ABSTAIN"]] = {
        "BULLISH": "BUY",
        "BEARISH": "SELL",
        "NEUTRAL": "NEUTRAL",
    }
    view = _MAP.get(overall, "ABSTAIN")
    return ProfessorAgentView(
        agent="technical",
        ticker=output.metadata.get("ticker", ""),
        view=view,
        confidence=output.confidence,
        rationale={"signals": output.metadata.get("signals", {})},
        risks=[],
        regime_label="",
        ts=_ts(output),
    )


def _adapt_sentiment(output: AgentOutput) -> ProfessorAgentView:
    bpct = float(output.metadata.get("stocktwits_bullish_pct", 50.0))
    if bpct > 55:
        view: Literal["BUY", "SELL", "NEUTRAL", "ABSTAIN"] = "BUY"
    elif bpct < 45:
        view = "SELL"
    else:
        view = "NEUTRAL"
    return ProfessorAgentView(
        agent="sentiment",
        ticker=output.metadata.get("ticker", ""),
        view=view,
        confidence=output.confidence,
        rationale={"stocktwits_bullish_pct": bpct},
        risks=[],
        regime_label="",
        ts=_ts(output),
    )


def _adapt_macro(output: AgentOutput) -> ProfessorAgentView:
    regime = output.metadata.get("regime", {})
    label = regime.get("label", "UNKNOWN")
    _MAP: dict[str, Literal["BUY", "SELL", "NEUTRAL", "ABSTAIN"]] = {
        "RISK_ON": "BUY",
        "RISK_OFF": "SELL",
        "RANGING": "NEUTRAL",
        "VOLATILE": "ABSTAIN",
    }
    view = _MAP.get(label, "ABSTAIN")
    return ProfessorAgentView(
        agent="macro",
        ticker=output.metadata.get("ticker", ""),
        view=view,
        confidence=output.confidence,
        rationale={"regime_label": label, "macro_snapshot": output.metadata.get("macro_snapshot", {})},
        risks=[],
        regime_label=label,
        ts=_ts(output),
    )


def _adapt_positioning(output: AgentOutput) -> ProfessorAgentView:
    sentiment = output.metadata.get("cboe_pc_sentiment", "")
    if sentiment == "CALL_DOMINANT":
        view: Literal["BUY", "SELL", "NEUTRAL", "ABSTAIN"] = "BUY"
    elif sentiment == "PUT_DOMINANT":
        view = "SELL"
    else:
        view = "NEUTRAL"
    return ProfessorAgentView(
        agent="positioning",
        ticker=output.metadata.get("ticker", ""),
        view=view,
        confidence=output.confidence,
        rationale={"cboe_pc_sentiment": sentiment, "pc_volume_ratio": output.metadata.get("pc_volume_ratio")},
        risks=[],
        regime_label="",
        ts=_ts(output),
    )


def _adapt_pelosi(output: AgentOutput) -> ProfessorAgentView:
    return ProfessorAgentView(
        agent="pelosi_agent",
        ticker=output.metadata.get("ticker", ""),
        view="ABSTAIN",
        confidence=0.0,
        rationale={},
        risks=[],
        regime_label="",
        ts=_ts(output),
    )


def _adapt_insider(output: AgentOutput) -> ProfessorAgentView:
    signals = output.metadata.get("top_signals", [])
    source = output.metadata.get("data_source", "")
    if len(signals) > 0 and source != "data_unavailable":
        view: Literal["BUY", "SELL", "NEUTRAL", "ABSTAIN"] = "BUY"
    else:
        view = "ABSTAIN"
    return ProfessorAgentView(
        agent="insider_buying",
        ticker=output.metadata.get("ticker", ""),
        view=view,
        confidence=output.confidence,
        rationale={"top_signals_count": len(signals), "data_source": source},
        risks=[],
        regime_label="",
        ts=_ts(output),
    )


def _adapt_options(output: AgentOutput) -> ProfessorAgentView:
    verdict = output.metadata.get("regime", {}).get("verdict", "")
    _MAP: dict[str, Literal["BUY", "SELL", "NEUTRAL", "ABSTAIN"]] = {
        "GREEN": "BUY",
        "RED": "SELL",
        "YELLOW": "NEUTRAL",
    }
    view = _MAP.get(verdict, "ABSTAIN")
    return ProfessorAgentView(
        agent="options_strategy",
        ticker=output.metadata.get("ticker", ""),
        view=view,
        confidence=output.confidence,
        rationale={"verdict": verdict},
        risks=[],
        regime_label="",
        ts=_ts(output),
    )


_ADAPTERS: dict[str, Callable[[AgentOutput], ProfessorAgentView]] = {
    "fundamental": _adapt_fundamental,
    "technical": _adapt_technical,
    "sentiment": _adapt_sentiment,
    "macro": _adapt_macro,
    "positioning": _adapt_positioning,
    "pelosi_agent": _adapt_pelosi,
    "insider_buying": _adapt_insider,
    "options_strategy": _adapt_options,
}
_EXPECTED_AGENTS = frozenset(_ADAPTERS)


def adapt_all(
    graph_state: dict,
    ticker: str,
    regime_label: str,
) -> list[ProfessorAgentView]:
    """Adapt all agent outputs from graph_state into ProfessorAgentView list.

    Uses .get() on graph_state — never raises KeyError on absent agents.
    Injects regime_label into every view via model_copy().
    Silently skips None agents and logs warnings on adapter failures.
    """
    views: list[ProfessorAgentView] = []
    for agent_id, adapter_fn in _ADAPTERS.items():
        agent_output = graph_state.get(agent_id)
        if agent_output is None:
            continue
        try:
            view = adapter_fn(agent_output)
            updated = view.model_copy(update={"regime_label": regime_label})
            assert (
                updated.regime_label == regime_label
            ), "regime_label injection into ProfessorAgentView failed — field may have been renamed"
            views.append(updated)
        except Exception as e:
            logger.warning("adapter failed: agent=%s error=%s", agent_id, e)
    return views
