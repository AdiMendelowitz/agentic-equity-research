"""
Cost and signal quality thresholds for the pipeline.

These constants are the single source of truth for kill-switch logic, alerting,
and the investment-case decision tree.

No logic lives here - only named, typed, documented constants.
All monetary values are USD. All rates and fractions are in [0.0, 1.0] unless noted.

Usage:
    from aer.utils.cost_thresholds import MAX_COST_PER_SIGNAL_USD, KILL_SWITCH_PRECISION_FLOOR
"""

from typing import Final

# ---------------------------------------------------------------------------
# Signal quality gates
# ---------------------------------------------------------------------------

MIN_SIGNAL_PRECISION: Final[float] = 0.52
"""
Minimum acceptable directional precision of final synthesis recommendations
over a 30-day rolling window.

Trigger: if measured precision falls below this value, escalate to PIVOT review.
Action: pause deployment of new ticker briefs; audit synthesis prompt and agent
        outputs for systematic bias. Do not use signals for capital allocation.

Rationale: 0.52 is slightly above chance (0.50) but conservative enough to
           detect a broken pipeline while avoiding false alarms from short windows.
           A skilled active manager benchmark is typically 0.55+; we set the floor
           lower to catch catastrophic failure, not to define success.
"""

MIN_WEEKLY_SIGNAL_BASE_RATE: Final[int] = 3
"""
Minimum number of actionable briefs (BUY or SELL conviction >= 0.6) produced
per week across the watchlist.

Trigger: if weekly signal count falls below this for 2 consecutive weeks,
         pipeline is not generating tradeable output.
Action: expand watchlist, reduce conviction threshold, or investigate
        data-source failures causing HOLD-only output.

Rationale: 3 signals/week at 5-day cadence = ~1 signal per 1.67 trading days.
           Below this, the pipeline is not operationally useful for active management.
"""

MAX_SIGNAL_LAG_HOURS: Final[int] = 6
"""
Maximum acceptable wall-clock hours from data release event (e.g. earnings, FOMC)
to brief availability in the UI.

Trigger: if measured p95 lag exceeds this on any calendar week, flag as latency
         regression requiring pipeline profiling.
Action: profile graph.py node durations; check free-tier provider timeout
        (FREE_MODEL_TIMEOUT=15s) contribution to overall latency.

Rationale: signals older than 6 hours have materially lower alpha in momentum-
           driven event windows. Same-day is the operational target; 6h is the
           acceptable ceiling before information decays to market consensus.
"""

# ---------------------------------------------------------------------------
# Cost gates
# ---------------------------------------------------------------------------

MAX_COST_PER_SIGNAL_USD: Final[float] = 2.00
"""
Maximum acceptable all-in LLM API cost per actionable signal (one ticker brief).

Trigger: if measured cost/brief exceeds this on Anthropic fallback,
         free-tier routing has collapsed and must be investigated immediately.
Action: verify OpenRouter/NIM availability; check Ollama daemon health;
        review llm_config.py provider keys.

Rationale: at $2.00/signal and 3 signals/week, annual LLM cost = $312 —
           still below MAX_ANNUAL_LLM_COST_USD. At measured all-free-routing cost
           of ~$0.036/signal, this threshold has 55× headroom and will only
           trigger on sustained total free-tier failure.
"""

MAX_ANNUAL_LLM_COST_USD: Final[float] = 500.00
"""
Maximum acceptable annual Anthropic API spend across all pipeline components
(all pipeline agents combined).

Trigger: if trailing-12-month Anthropic spend exceeds this, the free-first
         routing is failing at a frequency that makes the architecture uneconomical.
Action: audit free-tier provider reliability; consider NIM paid tier if
        nemotron-120b:free has high rejection rates; evaluate Ollama model upgrades.

Rationale: $500/year is ~10× the realistic all-free-routing estimate ($13–44/year).
           Exceeding it indicates a routing failure, not a business-as-usual cost.
           Below $500 the Anthropic cost is noise relative to maintenance overhead.
"""

MAX_TWITTER_COST_PER_DAY_USD: Final[float] = 5.00
"""
Maximum acceptable Twitter/X API spend in any rolling 24-hour window.

Trigger: if daily cumulative Twitter API cost exceeds this value, halt Twitter calls
         for the remainder of the day and log a WARNING.
Action: suspend Twitter API calls; use cached sentiment data or skip section.

Rationale: Twitter API v2 enterprise tier at realistic access levels costs ~$0.50–$1.50/day.
           $5.00 provides 3–10× headroom and catches runaway API errors before
           unexpected charges materialize (cost control).
"""

# ---------------------------------------------------------------------------
# Kill-switch floors and multipliers
# ---------------------------------------------------------------------------

KILL_SWITCH_PRECISION_FLOOR: Final[float] = 0.45
"""
Hard precision floor triggering pipeline shutdown recommendation.

Trigger: if measured 30-day rolling precision falls below 0.45, the pipeline
         is generating sub-chance signals (directional accuracy below coin flip).
Action: immediately halt all capital allocation using pipeline signals;
        open a root-cause investigation covering: synthesis prompt integrity,
        agent data-source staleness, and model-tier routing degradation.
        Do not restart without a documented fix and fresh 14-day validation window.

Rationale: 0.45 is below 0.50 chance baseline — signals are actively misleading.
           A 7-point gap below MIN_SIGNAL_PRECISION (0.52) ensures this floor
           is only hit by sustained catastrophic failure, not statistical noise
           in a short evaluation window.
"""

KILL_SWITCH_COST_MULTIPLIER: Final[float] = 3.0
"""
Multiplier applied to MAX_ANNUAL_LLM_COST_USD to derive the hard shutdown threshold.

Kill-switch annual LLM cost = MAX_ANNUAL_LLM_COST_USD × KILL_SWITCH_COST_MULTIPLIER = $1,500

Trigger (annual LLM): if Anthropic trailing-12-month spend exceeds $1,500,
 shut down Anthropic-path routing entirely; revert to free-tier-only or Ollama-only.

Rationale: 3× provides one alert cycle (hit MAX_* → investigate → fix) before
 reaching the shutdown level. A 2× multiplier would leave no reaction
 window; a 5× multiplier would allow runaway spend before action.
"""

KILL_SWITCH_OLLAMA_UPTIME_FLOOR: Final[float] = 0.80
"""
Minimum acceptable Ollama daemon uptime fraction over a 7-day rolling window.

Trigger: if Ollama is_ollama_available returns False for >20% of health-checks
 in any 7-day window (i.e. uptime < 0.80), the local routing tier is
 considered degraded and all Ollama-primary calls fall through to paid
 Anthropic fallback at unbudgeted frequency.
Action: log a daily WARNING; evaluate whether qwen2.5:7b should be replaced
 with a lighter model (llama3.2:3b) for higher CPU availability; review
 whether the host restarts Ollama after system reboots.

Rationale: at 80% uptime, 20% of calls route to Anthropic. For agents
 with N-per-run loops, this translates directly to the Loop-over-N risk
 (20% fallback rate × 160 potential calls/day × $0.0023
 = $0.074/day → $27/year — still under MAX_ANNUAL_LLM_COST_USD but
 approaching it at scale. Below 0.80 uptime, Anthropic fallback cost dominates.
"""

MAX_LLM_CALLS_PER_AGENT_RUN: Final[int] = 25
"""
Hard ceiling on LLM insight-extraction calls per single alt-data agent run.

The alt-data agents loop over N items (posts / sectors) and make one
LLM call per item. Without a cap, an unusually large day (viral spike, mass-layoff
event) produces unbounded calls — the only Loop-over-N cost risk in the pipeline.

Trigger: once an agent run reaches this many extraction calls, it stops processing
 the remaining items for that run and logs a WARNING. Unprocessed items are
 picked up on the next scheduled run (the dedup gate prevents reprocessing
 already-handled items).
Default 60: covers 8 IG accounts × ~3 posts + headroom, and typical sector counts,
 while bounding worst-case daily spend well under MAX_COST_PER_SIGNAL_USD
 aggregate. Raise only with a documented cost projection.
"""

# ---------------------------------------------------------------------------
# Prefetch - interactive / on-demand fetch rate cap
# ---------------------------------------------------------------------------

MAX_INTERACTIVE_FETCHES_PER_SESSION: Final[int] = 25
"""
Per-session ceiling for off-watchlist / on-demand interactive fetches.

An "interactive fetch" is any prefetch triggered by a UI action for a ticker that is NOT
on the daily watchlist — e.g. a user-initiated ticker lookup or a one-off brief generation
for an ad-hoc ticker. Watchlist prefetch in the daily run is exempt and does not count
toward this cap.

Trigger: once the session accumulates this many interactive fetches, the circuit-breaker
 activates and marks the affected brief section "data unavailable" rather than
 silently rendering empty data as if it were a confident result. The breach is
 propagated into InvestmentBrief.data_freshness / degraded=True.
Action: the user sees a "data unavailable" badge for the capped section; the scheduler
 refills the warm cache on the next scheduled daily run, which is exempt from
 this cap. Do not raise this value without documenting the cost projection.

Rationale: mirrors MAX_TWITTER_COST_PER_DAY_USD in style and purpose — both are
 per-session/per-day abuse-control knobs with a circuit-breaker that degrades
 gracefully rather than running unbounded. 25 on-demand fetches per session
 covers realistic interactive usage (a user browsing ~10 tickers with multiple
 data domains each) while preventing runaway costs from automated scraping or
 a looping UI bug.
"""
