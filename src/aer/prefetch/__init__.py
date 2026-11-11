"""aer.prefetch: declarative daily prefetch manifest.

Public API
----------
SourceSpec frozen descriptor for one external data source
MANIFEST list[SourceSpec]: all external data domains
map_key_prefix_to_source cache-key to SourceSpec.name reverse lookup
new_run_id generate a unique run identifier (uuid4 hex)
record_completeness persist per-(ticker, domain) JSONL completeness manifest
effective_ttl runtime TTL resolver: market-hours short TTL + market_aware extension
is_market_hours NYSE weekday 09:30-16:00 ET predicate (now-injectable, shared util)
should_repull_same_day True for prices/quote specs during market hours same-day re-pull
InteractiveRateLimiter per-session counter; allow blocks at MAX_INTERACTIVE_FETCHES_PER_SESSION
mark_section_unavailable set brief.degraded=True + data_freshness[section]="data unavailable"
"""

from aer.prefetch.circuit_breaker import InteractiveRateLimiter, mark_section_unavailable
from aer.prefetch.completeness import new_run_id, record_completeness
from aer.prefetch.manifest import MANIFEST, SourceSpec, map_key_prefix_to_source
from aer.prefetch.market_aware import effective_ttl, is_market_hours, should_repull_same_day

__all__ = [
    "MANIFEST",
    "SourceSpec",
    "map_key_prefix_to_source",
    "new_run_id",
    "record_completeness",
    "effective_ttl",
    "InteractiveRateLimiter",
    "is_market_hours",
    "mark_section_unavailable",
    "should_repull_same_day",
]
