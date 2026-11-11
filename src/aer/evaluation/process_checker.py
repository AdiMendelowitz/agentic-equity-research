"""Structural quality checker for InvestmentBrief. No LLM calls, no network calls."""

from urllib.parse import urlparse

from aer.models import InvestmentBrief


def _sources_appear_valid(sources: list[str]) -> bool:
    """Return True if every source URL has a non-empty scheme and netloc."""
    if not sources:
        return False
    for s in sources:
        try:
            parsed = urlparse(s)
            if not parsed.scheme or not parsed.netloc:
                return False
        except Exception:
            return False
    return True


def check_structural_quality(brief: InvestmentBrief) -> dict[str, bool]:
    """Return a flat dict of structural quality checks. All pure Python — no I/O."""
    agent_fields = [brief.fundamental, brief.technical, brief.sentiment, brief.macro, brief.positioning]
    populated = [a for a in agent_fields if a is not None]

    agents_complete = all(bool(a.content) and 0.0 <= a.confidence <= 1.0 for a in populated)

    all_sources: list[str] = []
    for a in populated:
        all_sources.extend(a.sources)

    sources_valid = _sources_appear_valid(all_sources) if all_sources else False

    all_agree = False
    if populated:
        direction_votes = []
        for a in [brief.fundamental, brief.sentiment, brief.positioning]:
            if a is not None and a.confidence > 0.65:
                direction_votes.append("UP" if brief.recommendation in ("BUY",) else "DOWN")
        all_agree = len(set(direction_votes)) <= 1 and len(direction_votes) > 0

    return {
        "agents_complete": agents_complete,
        "sources_appear_valid": sources_valid,
        "ml_prediction_present": brief.ml_prediction is not None,
        "backtest_present": brief.backtesting is not None,
        "all_agents_agree_direction": all_agree,
    }
