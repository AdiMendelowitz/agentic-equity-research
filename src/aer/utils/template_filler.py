"""Template filling utility: deterministic placeholder substitution."""

import logging
import re
import threading
from dataclasses import dataclass, field

from aer.paths import RESOURCES_DIR

logger = logging.getLogger(__name__)


_TEMPLATES_PATH = RESOURCES_DIR / "commentary_templates.md"
_template_lock = threading.Lock()
_template_cache: dict[str, str] = {}
_template_mtime: float = 0.0


def _load_templates() -> dict[str, str]:
    """Parse COMMENTARY_TEMPLATES.md and return {TMPL_KEY: template_text} dict."""
    text = _TEMPLATES_PATH.read_text(encoding="utf-8")
    matches = re.findall(r"### (TMPL_\w+)\n```\n([\s\S]*?)\n```", text)
    return {key: body.strip() for key, body in matches}


def get_template_text(key: str) -> str | None:
    """Return template text for key, or None if key not found. Invalidates cache on file change."""
    global _template_mtime, _template_cache
    if not _TEMPLATES_PATH.exists():
        logger.warning("get_template_text: COMMENTARY_TEMPLATES.md not found at %s", _TEMPLATES_PATH)
        return None
    with _template_lock:
        mtime = _TEMPLATES_PATH.stat().st_mtime
        if mtime != _template_mtime:
            _template_cache = _load_templates()
            _template_mtime = mtime
    return _template_cache.get(key)


@dataclass
class TemplateResult:
    """Result of a template fill attempt.

    filled is None iff ANY placeholder is absent from context.
    missing_keys is always a list; populated on None results.
    """

    filled: str | None
    missing_keys: list[str] = field(default_factory=list)
    template_key: str = ""


def build_context(
    views: list,
    macro: dict,
    risk: dict,
    conflicts: list[dict],
) -> dict:
    """Build a flat context dict from agent views, macro snapshot, risk dict, and conflicts.

    Extracts the most common placeholder values used across COMMENTARY_TEMPLATES.md.
    Never raises — missing fields produce empty strings.
    """
    ctx: dict = {}
    # Ticker: first non-empty ticker from views
    for v in views:
        ticker = v.ticker if hasattr(v, "ticker") else v.get("ticker", "")
        if ticker:
            ctx["ticker"] = ticker
            break
    ctx.setdefault("ticker", "")

    # Macro regime
    ctx["macro_regime"] = macro.get("regime_label", macro.get("label", ""))
    ctx["regime"] = ctx["macro_regime"]

    # Risk
    ctx["risk_score"] = str(risk.get("score", ""))

    # Views from named agents
    for v in views:
        agent = v.agent if hasattr(v, "agent") else v.get("agent", "")
        view_val = v.view if hasattr(v, "view") else v.get("view", "")
        conf = v.confidence if hasattr(v, "confidence") else v.get("confidence", 0.0)
        rationale = v.rationale if hasattr(v, "rationale") else v.get("rationale", {})
        if agent == "fundamental":
            ctx["valuation_label"] = str(rationale.get("valuation", ""))
        if agent == "technical":
            ctx["technical_signal"] = view_val
        if agent == "sentiment":
            ctx["sentiment_score"] = str(rationale.get("stocktwits_bullish_pct", ""))
        if agent == "positioning":
            ctx["call_flow_summary"] = str(rationale.get("cboe_pc_sentiment", ""))
        if agent == "insider_buying":
            n = rationale.get("top_signals_count", 0)
            ctx["insider_summary"] = f"{n} insider signals" if n else "no recent signals"
        if agent == "pelosi_agent":
            ctx["pelosi_view"] = view_val
        ctx[f"{agent}_id"] = agent
        ctx[f"{agent}_view"] = view_val
        ctx[f"{agent}_confidence"] = conf
        ctx[f"{agent}_confidence_pct"] = str(round(conf * 100))

    # First conflict
    if conflicts:
        top = conflicts[0]
        ctx["conflict_pattern"] = top.get("pattern", "")
        ctx["main_caveat"] = top.get("pattern", "")
        top_views = top.get("views", [])
        if top_views:
            tv = top_views[0]
            ctx["agent_id"] = tv.get("agent", "") if isinstance(tv, dict) else getattr(tv, "agent", "")
            ctx["view"] = tv.get("view", "") if isinstance(tv, dict) else getattr(tv, "view", "")
            conf_raw = tv.get("confidence", 0.0) if isinstance(tv, dict) else getattr(tv, "confidence", 0.0)
            ctx["confidence_pct"] = str(round(float(conf_raw) * 100))
        ctx["n_aligned"] = str(len(top_views))
        ctx["risk_score_val"] = str(top.get("details", {}).get("risk_score", ""))
    ctx.setdefault("conflict_pattern", "")
    ctx.setdefault("main_caveat", "")

    # Overview placeholders
    regime_desc = {
        "RISK_ON": "Markets are in a risk-on regime",
        "RISK_OFF": "Markets are in a risk-off regime",
        "RANGING": "Markets are in a ranging regime",
        "VOLATILE": "Markets are in a volatile regime",
    }
    ctx["macro_regime_sentence"] = regime_desc.get(ctx["macro_regime"], f"Current regime: {ctx['macro_regime']}")
    risk_score_int = 0
    try:
        risk_score_int = int(float(ctx["risk_score"]))
    except (ValueError, TypeError):
        pass
    if risk_score_int >= 7:
        ctx["risk_sentence"] = f"Risk score is elevated at {ctx['risk_score']}/10"
    elif risk_score_int >= 4:
        ctx["risk_sentence"] = f"Risk score is moderate at {ctx['risk_score']}/10"
    else:
        ctx["risk_sentence"] = f"Risk score is low at {ctx['risk_score']}/10"
    ctx.setdefault("valuation_sentence", ctx.get("valuation_label", "Valuation data unavailable"))
    ctx["valuation_sentence"] = ctx.get("valuation_label", "Valuation data unavailable") or "Valuation data unavailable"

    return ctx


def fill_template(template: str, context: dict, template_key: str = "", regime_tag: str = "") -> TemplateResult:
    """Fill template placeholders from context dict.

    filled=None iff ANY placeholder is absent from context.
    missing_keys is always populated on None result.
    Never raises — logs warning on None result.

    When regime_tag and template_key are given, a regime-specific variant
    ({template_key}_{REGIME_TAG}) is used if authored; otherwise the generic template is
    used (variants exist only for MACRO_OVERRIDE and VALUATION_MOMENTUM at present).
    """
    # Regime-variant override: try {template_key}_{REGIME_TAG} first; fall back to the generic
    # template on miss. Authored regime variants currently exist only for MACRO_OVERRIDE and
    # VALUATION_MOMENTUM (see COMMENTARY_TEMPLATES.md). For any other template family this lookup
    # misses and the generic template is used — an intentional, documented no-op, logged at DEBUG
    # so the fallback is observable rather than silent.
    if regime_tag and template_key:
        variant_key = f"{template_key}_{regime_tag.upper()}"
        variant_text = get_template_text(variant_key)
        if variant_text is not None:
            template = variant_text  # use regime variant; rest of function unchanged
        else:
            logger.debug(
                "fill_template: no regime variant %r for template_key=%r — using generic template",
                variant_key,
                template_key,
            )
    placeholders = re.findall(r"\{(\w+)\}", template)
    missing: list[str] = [key for key in placeholders if key not in context]
    if missing:
        logger.warning(
            "fill_template: template_key=%r missing keys=%r",
            template_key,
            missing,
        )
        return TemplateResult(filled=None, missing_keys=missing, template_key=template_key)
    fill_ctx = {k: context[k] for k in placeholders}
    try:
        filled = template.format(**fill_ctx)
    except (KeyError, IndexError, ValueError) as e:
        logger.warning("fill_template: format error for template_key=%r: %s", template_key, e)
        return TemplateResult(filled=None, missing_keys=list(set(missing + [str(e)])), template_key=template_key)
    return TemplateResult(filled=filled, missing_keys=[], template_key=template_key)
