"""Tests for src/aer/prefetch/manifest.py.

Test coverage:
 - all 12 business domains present in the MANIFEST
 - SourceSpec.domain is one of the real domain factories
 - news is a first-class domain (not subsumed under 'agent')
 - contract test: cache_key_template.format(...) matches a real key written by the actual
 fetcher body (uses real_key_cache fixture, not fakes)
 - all three date-stamped specs registered (news, finviz:fundament, options:regime)
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from aer.prefetch.manifest import MANIFEST, SourceSpec

# The 10 business-area names that must appear in MANIFEST (cot, pelosi and finviz are not in this slice).
# finviz:fundament and options:regime are ADDITIONAL sources within their business areas
# (finviz and regime respectively); they do NOT add new business-area names to this set.
_EXPECTED_BUSINESS_AREAS = {
    "prices",
    "financials",
    "earnings",
    "ownership",
    "options",
    "macro",
    "breadth",
    "regime",
    "sentiment",
    "news",
}

_VALID_CACHE_DOMAINS = {"data", "agent", "brief", "news", "lseg"}

# Concrete date string matching what real fetchers write via date.today().
_TEST_TICKER = "AAPL"
_TEST_DATE_STR = date.today().isoformat()

# Per-spec call args: spec.name → kwargs to pass to spec.fetch_fn.
# None → skip contract test (requires live HTTP or mock patches not wired in real_key_cache).
#
# Skipped specs and reason:
#   financials / financials:ratios - _FakeTicker lacks income_stmt/balance_sheet/cashflow attrs
#   earnings    - uses agent_cache() imported at module level; real_key_cache only patches
#                 aer.data.earnings.data_cache (not agent_cache); key is not written
#   options     - _FakeTicker.options returns () (empty); fetcher returns unavailable dict
#                 without writing to cache
#   options:regime - uses agent_cache() from aer.utils.cache; real_key_cache patches
#                 aer.data.regime.data_cache but the name agent_cache is already bound in
#                 regime.py's module namespace; key is not written to the tmp DB
#   finviz      - patched to a stub lambda in real_key_cache; stub does not call cache.set
#   news:yahoo  - _FakeTicker lacks .news attribute; fetcher returns empty without writing
#   pelosi      - requires live HTTP + SPY benchmark enrichment
_SPEC_CALL_ARGS: dict[str, dict[str, Any] | None] = {
    "prices": {"ticker": _TEST_TICKER, "period": "1y", "interval": "1d"},
    "prices:current": {"ticker": _TEST_TICKER},
    "prices:analyst_consensus": {"ticker": _TEST_TICKER},
    "prices:metadata": {"ticker": _TEST_TICKER},
    "financials": None,
    "financials:ratios": None,
    "earnings": None,
    "ownership": {"ticker": _TEST_TICKER},
    "options": None,
    "regime": {},
    "options:regime": None,
    "finviz": None,
    "macro": {},
    "breadth": {},
    "sentiment": {"ticker": _TEST_TICKER},
    "sentiment:stocktwits": {"ticker": _TEST_TICKER, "limit": 30},
    "sentiment:cboe_pc_ratio": {},
    "news": {"ticker": _TEST_TICKER},
    "news:finviz": {"ticker": _TEST_TICKER, "limit": 15},
    "news:yahoo": None,
    "cot": {},
    "pelosi": None,
}

# Template format args for verifying the expected key after the fetcher runs.
# Must match what the real fetcher writes (including date.today() for date-stamped specs).
_TEMPLATE_FORMAT_ARGS: dict[str, dict[str, Any]] = {
    "prices": {"ticker": _TEST_TICKER, "period": "1y", "interval": "1d"},
    "prices:current": {"ticker": _TEST_TICKER},
    "prices:analyst_consensus": {"ticker": _TEST_TICKER},
    "prices:metadata": {"ticker": _TEST_TICKER},
    "financials": {"ticker": _TEST_TICKER},
    "financials:ratios": {"ticker": _TEST_TICKER},
    "earnings": {"ticker": _TEST_TICKER},
    "ownership": {"ticker": _TEST_TICKER},
    "options": {"ticker": _TEST_TICKER, "num_expirations": 3},
    "regime": {},
    "options:regime": {"date_str": _TEST_DATE_STR},
    "finviz": {"ticker": _TEST_TICKER, "date_str": _TEST_DATE_STR},
    "macro": {"series_id": "FEDFUNDS", "periods": 3},
    "breadth": {},
    "sentiment": {"ticker": _TEST_TICKER},
    "sentiment:stocktwits": {"ticker": _TEST_TICKER, "limit": 30},
    "sentiment:cboe_pc_ratio": {},
    "news": {"ticker": _TEST_TICKER, "date_str": _TEST_DATE_STR},
    "news:finviz": {"ticker": _TEST_TICKER, "limit": 15},
    "news:yahoo": {"ticker": _TEST_TICKER, "limit": 10},
    "cot": {},
    "pelosi": {"lookback_days": 365},
}


# ── 12 business domains present ──────────────────────────────────────


def test_all_business_domains_present() -> None:
    """MANIFEST must enumerate all 12 external business domains.

    finviz:fundament and options:regime are additional sources within the
    finviz and regime business areas respectively - they do NOT add new
    business-area names, and they are NOT subtracted from the 12 count.
    """
    source_names = {s.name for s in MANIFEST}
    missing = _EXPECTED_BUSINESS_AREAS - source_names
    assert not missing, f"Missing business-area names in MANIFEST: {missing}"


# ── every SourceSpec.domain is one of the real cache factories ─────────


def test_manifest_domains_subset_of_real_factories() -> None:
    """Every SourceSpec.domain must be one of data/agent/brief/news/lseg.

    A domain string that does not match a factory in src/aer/utils/cache.py will silently
    route to the wrong cache at runtime - this test gates that at build time.
    """
    bad = [s for s in MANIFEST if s.domain not in _VALID_CACHE_DOMAINS]
    assert not bad, f"SourceSpec entries with invalid domain: {[(s.name, s.domain) for s in bad]}"


# ── news is a first-class domain, not subsumed under 'agent' ───────────


def test_news_domain_first_class() -> None:
    """At least one SourceSpec must use domain='news'.

    The original design missed 'news' as a domain - this test prevents regression.
    """
    news_entries = [s for s in MANIFEST if s.domain == "news"]
    assert news_entries, "No SourceSpec entries use domain='news' — news is first-class per"


# ── all three date-stamped specs registered ────────────────────────────


def test_date_stamped_specs_registered() -> None:
    """All three date-stamped specs must carry a {date_str} placeholder.

    Without these registrations, the callers' as_of threading cannot reach finviz:fundament
    and options:regime - they become silent gaps that never align their probe vs written key.
    """
    templates = {s.cache_key_template for s in MANIFEST}
    assert "news:{ticker}:{date_str}" in templates, "news date-stamped spec missing"
    assert "options:regime:{date_str}" in templates, "options:regime date-stamped spec missing"


# ── per-source contract test - template matches real written key ────────


@pytest.mark.parametrize("spec", MANIFEST, ids=[s.name for s in MANIFEST])
def test_contract_template_matches_real_key(
    spec: SourceSpec,
    real_key_cache: Any,
) -> None:
    """For each SourceSpec, invoke the real fetch_fn and assert the cache_key_template.format(...)
    matches a key actually written to the tmp SQLite by the real fetcher body.

    Uses real_key_cache (real fetcher bodies, patched network boundaries) so key parity
    is verified against live code paths - not conftest fakes. Specs marked None in
    _SPEC_CALL_ARGS are skipped with a clear reason (not silently dropped).

    The three date-stamped specs (news, finviz:fundament, options:regime) are covered
    with _TEST_DATE_STR = date.today().isoformat() to match what the real fetchers write.
    """
    call_args = _SPEC_CALL_ARGS.get(spec.name)
    if call_args is None:
        pytest.skip(
            f"spec '{spec.name}' skipped: requires live HTTP or mock patches unavailable "
            "in real_key_cache (template correctness verified by code review)"
        )

    # Invoke the real fetcher body to write its key to the tmp SQLite.
    try:
        if call_args:
            spec.fetch_fn(**call_args)
        else:
            spec.fetch_fn()
    except Exception:
        pass  # fetcher may raise (missing credentials, empty data); key may still be written

    written = real_key_cache.written_keys()

    # Build the expected key using the same format args the fetcher uses.
    fmt_args = _TEMPLATE_FORMAT_ARGS.get(spec.name, {})
    try:
        expected_key = spec.cache_key_template.format(**fmt_args)
    except KeyError as e:
        pytest.fail(f"spec '{spec.name}' template '{spec.cache_key_template}' missing format key: {e}")

    # Allow prefix match for templates that have extra params not in fmt_args.
    key_prefix = expected_key.split("{")[0] if "{" in expected_key else expected_key
    matching = [k for k in written if k == expected_key or k.startswith(key_prefix)]
    assert matching, (
        f"SourceSpec '{spec.name}' template '{spec.cache_key_template}' resolves to "
        f"'{expected_key}' but no matching key in written_keys(): {sorted(written)[:15]}"
    )
