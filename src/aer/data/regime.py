"""Market regime classifier using macro snapshot data.

Five regimes based on threshold rules applied to VIX, credit spreads,
yield curve, inflation, and net Fed liquidity. Designed to run on whatever
subset of the macro snapshot is available — gracefully degrades if data is missing.

Usage:
    from aer.data.regime import classify_regime, get_regime
    regime = get_regime()   # uses live macro snapshot, cached 24h
"""

import logging
from datetime import date
from typing import Any

import pandas as pd

from aer.data.macro import get_macro_snapshot
from aer.data.prices import get_price_history
from aer.utils.cache import agent_cache, data_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

REGIME_DESCRIPTIONS: dict[str, str] = {
    "BULL_STEADY": "Risk-on expansion with low volatility. Accommodative conditions for equities.",
    "BULL_VOLATILE": "Risk-on bias but elevated uncertainty. Selectivity matters; quality premium.",
    "BEAR": "Risk-off. Defensive positioning warranted. Credit stress and/or liquidity withdrawal.",
    "STAGFLATION": "High inflation with slowing growth. Commodities and TIPS outperform. Equity P/E compression risk.",
    "RECOVERY": "Improving from trough. Early-cycle dynamics: beaten-down cyclicals and small-caps lead.",
    "UNKNOWN": "Insufficient data to classify regime reliably.",
}

REGIME_EQUITY_BIAS: dict[str, str] = {
    "BULL_STEADY": "BULLISH",
    "BULL_VOLATILE": "CAUTIOUSLY_BULLISH",
    "BEAR": "BEARISH",
    "STAGFLATION": "NEUTRAL_TO_BEARISH",
    "RECOVERY": "BULLISH",
    "UNKNOWN": "NEUTRAL",
}


def classify_regime(snap: dict[str, Any]) -> dict[str, Any]:
    """Classify regime from a macro snapshot dict.

    Scoring: each indicator votes +1 (bullish), -1 (bearish), or 0 (neutral).
    Special conditions (STAGFLATION, RECOVERY) are checked before the generic score.

    Args:
        snap: output of get_macro_snapshot() or any compatible dict.

    Returns:
        dict with regime, description, equity_bias, score, signals.
    """
    vix = snap.get("vix")
    yield_curve = snap.get("yield_curve")
    hy_spread = snap.get("hy_spread")
    cpi_yoy = snap.get("cpi_yoy_pct")
    unemployment = snap.get("unemployment")
    net_liq_trend = snap.get("net_liquidity_trend")
    dxy_chg = snap.get("dxy_change_pct")

    signals: list[str] = []
    score: int = 0

    # --- VIX ---
    if vix is not None:
        if vix < 18:
            score += 1
            signals.append(f"VIX {vix:.1f} — low fear (bullish)")
        elif vix < 25:
            signals.append(f"VIX {vix:.1f} — moderate (neutral)")
        elif vix < 35:
            score -= 1
            signals.append(f"VIX {vix:.1f} — elevated fear (bearish)")
        else:
            score -= 2
            signals.append(f"VIX {vix:.1f} — extreme fear (strongly bearish)")

    # --- Yield curve ---
    if yield_curve is not None:
        if yield_curve > 0.5:
            score += 1
            signals.append(f"Yield curve {yield_curve:+.2f}% — normal slope (bullish)")
        elif yield_curve > -0.3:
            signals.append(f"Yield curve {yield_curve:+.2f}% — flat (neutral)")
        elif yield_curve > -1.0:
            score -= 1
            signals.append(f"Yield curve {yield_curve:+.2f}% — inverted (bearish)")
        else:
            score -= 2
            signals.append(f"Yield curve {yield_curve:+.2f}% — deeply inverted (strongly bearish)")

    # --- HY credit spread ---
    if hy_spread is not None:
        if hy_spread < 350:
            score += 1
            signals.append(f"HY spread {hy_spread:.0f}bps — tight (bullish)")
        elif hy_spread < 500:
            signals.append(f"HY spread {hy_spread:.0f}bps — normal (neutral)")
        elif hy_spread < 700:
            score -= 1
            signals.append(f"HY spread {hy_spread:.0f}bps — widening (bearish)")
        else:
            score -= 2
            signals.append(f"HY spread {hy_spread:.0f}bps — distressed (strongly bearish)")

    # --- Net Fed liquidity ---
    if net_liq_trend is not None:
        if net_liq_trend == "expanding":
            score += 1
            signals.append("Net Fed liquidity expanding (bullish)")
        elif net_liq_trend == "contracting":
            score -= 1
            signals.append("Net Fed liquidity contracting (bearish)")
        else:
            signals.append("Net Fed liquidity neutral")

    # --- DXY ---
    if dxy_chg is not None:
        if dxy_chg > 0.5:
            score -= 1
            signals.append(f"DXY +{dxy_chg:.2f}% — dollar strengthening (risk-off headwind)")
        elif dxy_chg < -0.5:
            score += 1
            signals.append(f"DXY {dxy_chg:.2f}% — dollar weakening (risk-on tailwind)")

    # --- Special regime overrides ---

    # STAGFLATION: high inflation + weakening growth signals
    stagflation = (
        cpi_yoy is not None
        and cpi_yoy > 5.0
        and (
            (yield_curve is not None and yield_curve < 0)
            or (unemployment is not None and unemployment > 5.0)
            or (hy_spread is not None and hy_spread > 400)
        )
    )

    # RECOVERY: VIX falling from elevated, credit spreads tightening, yield curve re-steepening
    # Approximate: VIX moderate but credit spreads tightening (requires trend data — use net_liq)
    recovery = (
        vix is not None and 20 <= vix <= 30 and net_liq_trend == "expanding" and (hy_spread is None or hy_spread < 500)
    )

    # BEAR: strong negative signals
    bear = score <= -3 or (vix is not None and vix > 35) or (hy_spread is not None and hy_spread > 700)

    # Classify
    if stagflation:
        regime = "STAGFLATION"
    elif bear:
        regime = "BEAR"
    elif recovery:
        regime = "RECOVERY"
    elif score >= 2:
        if vix is not None and vix < 18:
            regime = "BULL_STEADY"
        else:
            regime = "BULL_VOLATILE"
    elif score >= 0:
        if vix is not None and vix < 22:
            regime = "BULL_STEADY"
        else:
            regime = "BULL_VOLATILE"
    elif score >= -2:
        # Score -2 (two mild bearish signals) is labeled BULL_VOLATILE rather than BEAR
        # because the bear flag (score <= -3) has not triggered; the regime is cautious
        # but not a confirmed bear. BULL_VOLATILE signals elevated caution without
        # contradicting the non-bear classification from the explicit bear check above.
        regime = "BULL_VOLATILE"
    else:
        regime = "BEAR"

    # Fall back to UNKNOWN if fewer than 2 indicators contributed
    if len(signals) < 2:
        regime = "UNKNOWN"

    return {
        "regime": regime,
        "description": REGIME_DESCRIPTIONS[regime],
        "equity_bias": REGIME_EQUITY_BIAS[regime],
        "score": score,
        "signals": signals,
    }


def _compute_vix_regime() -> dict[str, Any]:
    """Compute VIX/IV regime dict from current VIX price history.

    Returns dict with keys: vix_current, vix_52w_low, vix_52w_high, iv_rank, vix_regime.
    Raises on data unavailability.
    """

    df = get_price_history("^VIX", period="1y", interval="1d")
    if df is None or df.empty:
        raise RuntimeError("VIX price history unavailable")
    close = df["Close"]
    if hasattr(close, "iloc"):
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
    if close.empty:
        raise RuntimeError("VIX close series is empty")

    vix_current = float(close.iloc[-1])
    vix_52w_low = float(close.min())
    vix_52w_high = float(close.max())
    iv_rank = (
        round((vix_current - vix_52w_low) / (vix_52w_high - vix_52w_low) * 100, 1)
        if vix_52w_high != vix_52w_low
        else 50.0
    )

    if vix_current >= 30.0:
        vix_regime = "BACKWARDATION_SPIKE"
    elif vix_current >= 20.0:
        vix_regime = "ELEVATED_VIX"
    else:
        vix_regime = "CONTANGO_NORMAL"

    return {
        "vix_current": vix_current,
        "vix_52w_low": vix_52w_low,
        "vix_52w_high": vix_52w_high,
        "iv_rank": iv_rank,
        "vix_regime": vix_regime,
    }


def get_options_regime(*, as_of: date | None = None) -> dict[str, Any]:
    """Return cached VIX/IV regime for the current session.

    Cached at TTL_ANALYSIS (12h) — regime is global to the run, not per-ticker.
    as_of: run-level date for the cache key — prevents midnight-crossing
    key mismatch between prefetch write and downstream probe). Defaults to date.today.
    Key: options:regime:{as_of or date.today}
    Returns dict with keys: vix_current, vix_52w_low, vix_52w_high, iv_rank,
    vix_regime (one of CONTANGO_NORMAL, ELEVATED_VIX, BACKWARDATION_SPIKE).
    Returns empty dict on error.
    """
    stamp = as_of if as_of is not None else date.today()
    key = f"options:regime:{stamp}"
    cache = agent_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    try:
        regime = _compute_vix_regime()
    except Exception as exc:
        logger.warning("options regime computation failed: %s", exc)
        return {}
    cache.set(key, regime)
    return regime


def get_regime() -> dict[str, Any]:
    """Classify current market regime using live macro snapshot. Cached 24 h (domain='data')."""
    key = "regime:current"
    cache = data_cache()
    cached = cache.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    snap = get_macro_snapshot()
    result = classify_regime(snap)
    cache.set(key, result)
    return result
