"""Declarative source registry for the daily prefetch pass.

Each SourceSpec references an existing get_* fetcher by ref.
Adding a new source = append one entry. No other change required.

Domain Literal includes 'news' and 'lseg' as first-class values. A design with only
['data','agent','brief'] silently misrouted news-domain keys.

Forward-compatibility: market_aware and vintage fields are reserved
(per-source market-aware TTL and point-in-time vintage tracking). They default to
False and currently have no effect.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceSpec:
    """Frozen descriptor for one external data source in the prefetch manifest.

    Fields
    ------
    name:
    Friendly identifier used in gap logs and budget-violation warnings.
    Unique within MANIFEST.
    fetch_fn:
    Reference to the existing get_* fetcher (no rewrite).
    Fetcher owns its own cache.get/set and TTL internally.
    cache_key_template:
    The REAL cache key template the fetcher writes, with all params as
    named placeholders. Example: ``"ownership:{ticker}"``.
    Used by map_key_prefix_to_source and the contract test.
    ttl:
    Time-to-live in seconds; references a TTL_* constant from src/aer/data/cache.py.
    scope:
    ``"per-ticker"`` - fetcher is called once per watchlist ticker.
    ``"global"`` - fetcher is called once per prefetch pass.
    domain:
    Must be one of the real domain-cache factories in src/aer/utils/cache.py
    (data/agent/brief/news/lseg). Any other value silently misroutes
    to the wrong SQLite domain at runtime.
    call_budget:
    Per-domain external-call allowance from PREFETCH_BUDGET_* in
    src/aer/utils/constants.py. Driver enforces soft-skip at breach.
    market_aware:
    RESERVED: per-source market-aware TTL extension.
    Has no effect yet. Do not set True.
    vintage:
    RESERVED: point-in-time vintage / run manifest support.
    Has no effect yet. Do not set True.
    """

    name: str
    fetch_fn: Callable
    cache_key_template: str
    ttl: int
    scope: Literal["per-ticker", "global"]
    domain: Literal["data", "agent", "brief", "news", "lseg"]
    call_budget: int
    # Reserved: leave False in all SourceSpec entries
    market_aware: bool = field(default=False)
    vintage: bool = field(default=False)


def map_key_prefix_to_source(key: str) -> str | None:
    """Return the SourceSpec.name whose cache_key_template prefix matches `key`.

    Derives each spec's static prefix as the substring before the first ``{``.
    Returns None if no spec matches.

    This is a standalone module function (NOT a method on SourceSpec) so it can
    be called without a SourceSpec instance and does not pollute the frozen
    dataclass interface.
    """
    for spec in MANIFEST:
        prefix = spec.cache_key_template.split("{")[0]
        if key.startswith(prefix):
            return spec.name
    return None


# ---------------------------------------------------------------------------
# MANIFEST
# ---------------------------------------------------------------------------
# One SourceSpec per fetcher across all 12 external data domains.
# Real cache_key_template values are taken verbatim from the fetcher sources.
# TWO date-stamped specs are registered with {date_str} placeholders so the callers
# can thread a single run-level as_of date into both the prefetch write key and the
# downstream probe key:
#   1. news:{ticker}:{date_str}              domain=news,  per-ticker
#   2. options:regime:{date_str}             domain=agent, global
#
# IMPORTANT: `get_regime` (regime:current, domain=data) and `get_options_regime`
# (options:regime:{date_str}, domain=agent) are DISTINCT specs - both are registered.
#
# NOTE: prices:ohlcv is the split-adjusted path (display agents).
# prices:ohlcv_raw is the dividend-unadjusted modeling-facing entry, added
# with vintage=True. The display path (prices:ohlcv) keeps vintage=False.
# ---------------------------------------------------------------------------


def _build_manifest() -> list[SourceSpec]:
    """Build and return the full MANIFEST list.

    Imports are deferred to this function to avoid import-time side effects and
    circular dependencies (CLAUDE.md: no module-level date.today() or heavy imports).
    """
    # ── Breadth ──────────────────────────────────────────────────────────────
    from aer.data.breadth import get_breadth_snapshot  # noqa: PLC0415

    # ── Earnings ─────────────────────────────────────────────────────────────
    from aer.data.earnings import get_earnings_data  # noqa: PLC0415

    # ── Fama-French factors ────────────────────────────────────────
    from aer.data.factors import get_factor_data  # noqa: PLC0415

    # ── Financials ───────────────────────────────────────────────────────────
    from aer.data.financials import get_financial_statements, get_key_ratios  # noqa: PLC0415

    # ── Macro ────────────────────────────────────────────────────────────────
    from aer.data.macro import get_macro_snapshot  # noqa: PLC0415

    # ── News (date-stamped, per-ticker) ───────────────────────────────────────
    from aer.data.news_feeds import get_finviz_news, get_news_headlines, get_yahoo_news  # noqa: PLC0415

    # ── Options ──────────────────────────────────────────────────────────────
    from aer.data.options_flow import get_options_flow  # noqa: PLC0415

    # ── Ownership ────────────────────────────────────────────────────────────
    from aer.data.ownership import get_ownership  # noqa: PLC0415

    # ── Prices ────────────────────────────────────────────────────────────────
    from aer.data.prices import (  # noqa: PLC0415
        get_analyst_consensus,
        get_company_metadata,
        get_current_price,
        get_price_history,
        get_price_history_raw,
    )
    from aer.data.regime import get_options_regime, get_regime  # noqa: PLC0415

    # ── Sentiment ────────────────────────────────────────────────────────────
    from aer.data.sentiment_feeds import get_apewisdom, get_cboe_pc_ratio, get_stocktwits  # noqa: PLC0415
    from aer.utils.cache import (  # noqa: PLC0415
        TTL_ANALYSIS,
        TTL_NEWS,
        TTL_PRICES,
        TTL_QUOTE,
    )
    from aer.utils.constants import (  # noqa: PLC0415
        PREFETCH_BUDGET_BREADTH,
        PREFETCH_BUDGET_EARNINGS,
        PREFETCH_BUDGET_FACTORS,
        PREFETCH_BUDGET_FINANCIALS,
        PREFETCH_BUDGET_MACRO,
        PREFETCH_BUDGET_NEWS,
        PREFETCH_BUDGET_OPTIONS,
        PREFETCH_BUDGET_OWNERSHIP,
        PREFETCH_BUDGET_PRICES,
        PREFETCH_BUDGET_REGIME,
        PREFETCH_BUDGET_SENTIMENT,
    )

    return [
        # ── prices ──────────────────────────────────────────────────────────
        # prices:ohlcv - split+dividend-adjusted path used by all display agents.
        # vintage=False: display cache; last-write-wins is acceptable here.
        # prices + prices:current are freshness-critical - market_aware=True routes TTL
        # through market_aware_ttl (weekend/holiday extension) and enables the short
        # market-hours TTL + same-day re-pull in effective_ttl / should_repull_same_day.
        SourceSpec(
            name="prices",
            fetch_fn=get_price_history,
            cache_key_template="prices:ohlcv:{ticker}:{period}:{interval}",
            ttl=TTL_PRICES,
            scope="per-ticker",
            domain="data",
            call_budget=PREFETCH_BUDGET_PRICES,
            market_aware=True,
            vintage=False,
        ),
        # prices:ohlcv_raw - split-adjusted, dividend-UNADJUSTED modeling-facing entry.
        # vintage=True so the append-only vintage store gates this spec.
        # Raw prices for signal computation avoid look-ahead from
        # retroactive dividend rescaling (see get_price_history_raw docstring).
        SourceSpec(
            name="prices:ohlcv_raw",
            fetch_fn=get_price_history_raw,
            cache_key_template="prices:ohlcv_raw:{ticker}:{period}:{interval}",
            ttl=TTL_PRICES,
            scope="per-ticker",
            domain="data",
            call_budget=PREFETCH_BUDGET_PRICES,
            market_aware=True,
            vintage=True,
        ),
        SourceSpec(
            name="prices:current",
            fetch_fn=get_current_price,
            cache_key_template="prices:current:{ticker}",
            ttl=TTL_QUOTE,
            scope="per-ticker",
            domain="data",
            call_budget=PREFETCH_BUDGET_PRICES,
            market_aware=True,
        ),
        # prices:analyst_consensus is a modeling-relevant key (consensus used in
        # backtesting entry signals) - vintage=True gates the append-only path.
        SourceSpec(
            name="prices:analyst_consensus",
            fetch_fn=get_analyst_consensus,
            cache_key_template="prices:analyst_consensus:{ticker}",
            ttl=TTL_PRICES,
            scope="per-ticker",
            domain="data",
            call_budget=PREFETCH_BUDGET_PRICES,
            market_aware=True,
            vintage=True,
        ),
        SourceSpec(
            name="prices:metadata",
            fetch_fn=get_company_metadata,
            cache_key_template="prices:metadata:{ticker}",
            ttl=TTL_PRICES,
            scope="per-ticker",
            domain="data",
            call_budget=PREFETCH_BUDGET_PRICES,
            market_aware=True,
        ),
        # ── financials ──────────────────────────────────────────────────────
        # both financials specs are modeling-relevant - vintage=True.
        SourceSpec(
            name="financials",
            fetch_fn=get_financial_statements,
            cache_key_template="financials:statements:{ticker}",
            ttl=TTL_PRICES,
            scope="per-ticker",
            domain="data",
            call_budget=PREFETCH_BUDGET_FINANCIALS,
            market_aware=True,
            vintage=True,
        ),
        SourceSpec(
            name="financials:ratios",
            fetch_fn=get_key_ratios,
            cache_key_template="financials:ratios:{ticker}",
            ttl=TTL_PRICES,
            scope="per-ticker",
            domain="data",
            call_budget=PREFETCH_BUDGET_FINANCIALS,
            market_aware=True,
            vintage=True,
        ),
        # ── earnings ────────────────────────────────────────────────────────
        SourceSpec(
            name="earnings",
            fetch_fn=get_earnings_data,
            cache_key_template="earnings:{ticker}",
            ttl=TTL_PRICES,
            scope="per-ticker",
            domain="data",
            call_budget=PREFETCH_BUDGET_EARNINGS,
            market_aware=True,
        ),
        # ── ownership ───────────────────────────────────────────────────────
        # ownership is modeling-relevant (institutional concentration) - vintage=True.
        SourceSpec(
            name="ownership",
            fetch_fn=get_ownership,
            cache_key_template="ownership:{ticker}",
            ttl=TTL_ANALYSIS,
            scope="per-ticker",
            domain="data",
            call_budget=PREFETCH_BUDGET_OWNERSHIP,
            market_aware=True,
            vintage=True,
        ),
        # ── options flow ────────────────────────────────────────────────────
        SourceSpec(
            name="options",
            fetch_fn=get_options_flow,
            cache_key_template="options:flow:{ticker}:{num_expirations}",
            ttl=TTL_ANALYSIS,
            scope="per-ticker",
            domain="agent",
            call_budget=PREFETCH_BUDGET_OPTIONS,
            market_aware=True,
        ),
        # ── regime (non-date-stamped, global) ───────────────────────────────
        SourceSpec(
            name="regime",
            fetch_fn=get_regime,
            cache_key_template="regime:current",
            ttl=TTL_PRICES,
            scope="global",
            domain="data",
            call_budget=PREFETCH_BUDGET_REGIME,
            market_aware=True,
        ),
        # ── options regime (date-stamped, global) ─────────
        # DISTINCT from `regime` above. get_options_regime writes
        # options:regime:{date.today()} - register template as options:regime:{date_str}
        # so the caller threads as_of into the written key.
        SourceSpec(
            name="options:regime",
            fetch_fn=get_options_regime,
            cache_key_template="options:regime:{date_str}",
            ttl=TTL_ANALYSIS,
            scope="global",
            domain="agent",
            call_budget=PREFETCH_BUDGET_REGIME,
            market_aware=True,
        ),
        # ── macro ────────────────────────────────────────────────────────────
        # macro snapshot aggregates FRED rates/spreads (modeling-relevant
        # for regime filters) - vintage=True.
        SourceSpec(
            name="macro",
            fetch_fn=get_macro_snapshot,
            cache_key_template="macro:snapshot",
            ttl=TTL_PRICES,
            scope="global",
            domain="data",
            call_budget=PREFETCH_BUDGET_MACRO,
            market_aware=True,
            vintage=True,
        ),
        # ── breadth ─────────────────────────────────────────────────────────
        SourceSpec(
            name="breadth",
            fetch_fn=get_breadth_snapshot,
            cache_key_template="breadth:snapshot",
            ttl=TTL_PRICES,
            scope="global",
            domain="data",
            call_budget=PREFETCH_BUDGET_BREADTH,
            market_aware=True,
        ),
        # ── sentiment ───────────────────────────────────────────────────────
        SourceSpec(
            name="sentiment",
            fetch_fn=get_apewisdom,
            cache_key_template="sentiment:apewisdom:{ticker}",
            ttl=TTL_ANALYSIS,
            scope="per-ticker",
            domain="agent",
            call_budget=PREFETCH_BUDGET_SENTIMENT,
            market_aware=True,
        ),
        SourceSpec(
            name="sentiment:stocktwits",
            fetch_fn=get_stocktwits,
            cache_key_template="sentiment:stocktwits:{ticker}:{limit}",
            ttl=TTL_ANALYSIS,
            scope="per-ticker",
            domain="agent",
            call_budget=PREFETCH_BUDGET_SENTIMENT,
            market_aware=True,
        ),
        SourceSpec(
            name="sentiment:cboe_pc_ratio",
            fetch_fn=get_cboe_pc_ratio,
            cache_key_template="sentiment:cboe_pc_ratio",
            ttl=TTL_ANALYSIS,
            scope="global",
            domain="agent",
            call_budget=PREFETCH_BUDGET_SENTIMENT,
            market_aware=True,
        ),
        # ── news (domain='news', first-class) ─────────────────────────
        # market_aware intentionally False: TTL_NEWS is 1h - news goes stale fast
        # and market-aware TTL extension does not apply (src/aer/data/cache.py).
        SourceSpec(
            name="news",
            fetch_fn=get_news_headlines,
            cache_key_template="news:{ticker}:{date_str}",
            ttl=TTL_NEWS,
            scope="per-ticker",
            domain="news",
            call_budget=PREFETCH_BUDGET_NEWS,
        ),
        SourceSpec(
            name="news:finviz",
            fetch_fn=get_finviz_news,
            cache_key_template="finviz_news:{ticker}:{limit}",
            ttl=TTL_NEWS,
            scope="per-ticker",
            domain="news",
            call_budget=PREFETCH_BUDGET_NEWS,
        ),
        SourceSpec(
            name="news:yahoo",
            fetch_fn=get_yahoo_news,
            cache_key_template="yahoo_news:{ticker}:{limit}",
            ttl=TTL_NEWS,
            scope="per-ticker",
            domain="news",
            call_budget=PREFETCH_BUDGET_NEWS,
        ),
        # ── Fama-French factors ────────────────────────────────────
        SourceSpec(
            name="factors:ken_french_daily",
            fetch_fn=get_factor_data,
            cache_key_template="factors:ken_french_daily:{date_str}",
            ttl=TTL_PRICES,
            scope="global",
            domain="data",
            call_budget=PREFETCH_BUDGET_FACTORS,
            market_aware=False,
            vintage=False,
        ),
    ]


MANIFEST: list[SourceSpec] = _build_manifest()
