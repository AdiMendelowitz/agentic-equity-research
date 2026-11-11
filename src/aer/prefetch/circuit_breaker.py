"""interactive on-demand fetch rate cap and circuit-breaker.

Provides:
- InteractiveRateLimiter: per-session counter that blocks at MAX_INTERACTIVE_FETCHES_PER_SESSION.
 Instance-scoped - no process-global state (avoids the cross-caller bleed the 30a review flagged).
- mark_section_unavailable(brief, section): sets brief.degraded=True and
 brief.data_freshness[section]="data unavailable" so consumers see an explicit unavailable
 state rather than silently-empty confident data.
- apply_interactive_cap(brief, session_fetch_count, cap, domain): functional helper used by
 tests - returns a modified brief with degraded marker when session_fetch_count > cap.

Design notes:
- InteractiveRateLimiter counts off-watchlist / on-demand fetches only.
 Watchlist daily_runner prefetch is exempt and does NOT count against this cap.
- Instance-scoped: each calling context constructs its own limiter instance so
 different UI sessions / test cases do not share state (no module-level counter).
- The "data unavailable" sentinel populates the data_freshness/degraded field
 that InvestmentBrief already carries - populates it on a cap breach.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

# Sentinel value written to data_freshness on a circuit-breaker trip.
DATA_UNAVAILABLE_SENTINEL: str = "data-unavailable"


def mark_section_unavailable(
    brief: InvestmentBrief,  # type: ignore[name-defined]  # noqa: F821
    section: str,
) -> None:
    """Mark a brief section as data-unavailable in-place.

    Sets brief.degraded=True and brief.data_freshness[section]="data unavailable".
    Populates the InvestmentBrief.data_freshness/degraded fields so
    consumers render an explicit unavailable state, never silent-empty confident data.

    Parameters
    ----------
    brief:
        InvestmentBrief instance to mutate.
    section:
        Domain name / section key (e.g. "prices", "macro", "sentiment").
    """
    brief.degraded = True
    if brief.data_freshness is None:
        brief.data_freshness = {}
    brief.data_freshness[section] = DATA_UNAVAILABLE_SENTINEL
    _logger.warning(
        "circuit_breaker: section marked data-unavailable section=%s degraded=True",
        section,
    )


def apply_interactive_cap(
    brief: InvestmentBrief,  # type: ignore[name-defined]  # noqa: F821
    *,
    session_fetch_count: int,
    cap: int,
    domain: str,
) -> InvestmentBrief:  # type: ignore[name-defined]  # noqa: F821
    """Functional circuit-breaker helper for tests and the interactive fetch path.

    When session_fetch_count exceeds cap (inclusive breach), marks the named
    domain section as data-unavailable on the brief (degraded=True,
    data_freshness[domain]="data unavailable").

    Under cap: returns brief unchanged (degraded stays False).

    Parameters
    ----------
    brief:
        InvestmentBrief instance (modified in place; also returned for chaining).
    session_fetch_count:
        Number of interactive fetches accumulated this session, including the current one.
    cap:
        Per-session ceiling (typically MAX_INTERACTIVE_FETCHES_PER_SESSION).
    domain:
        The brief section domain name to mark unavailable on a breach.

    Returns
    -------
    The same InvestmentBrief instance (mutated if cap exceeded).
    """
    if session_fetch_count > cap:
        mark_section_unavailable(brief, domain)
        _logger.warning(
            "circuit_breaker: interactive cap exceeded session_fetch_count=%d cap=%d domain=%s",
            session_fetch_count,
            cap,
            domain,
        )
    return brief


class InteractiveRateLimiter:
    """Per-session counter that gates off-watchlist on-demand fetches.

    Instance-scoped: each calling context constructs its own instance so there is
    no cross-caller state bleed (the 30a review's concern about module-level mutable
    state is avoided by design).

    Usage
    -----
        limiter = InteractiveRateLimiter()
        if limiter.allow():
            result = fetch_data(ticker)
        else:
            mark_section_unavailable(brief, "prices")

    The count increments on every allow() call (whether or not the caller actually
    fetches).  allow() returns False at the inclusive breach point (count >= cap),
    ensuring exact boundary match with the boundary-test helper contract.
    """

    def __init__(self, cap: int | None = None) -> None:
        """Initialise the limiter.

        Parameters
        ----------
        cap:
            Per-session ceiling.  Defaults to MAX_INTERACTIVE_FETCHES_PER_SESSION
            at construction time (never at module import time - CLAUDE.md).
        """
        if cap is None:
            from aer.utils.cost_thresholds import MAX_INTERACTIVE_FETCHES_PER_SESSION  # noqa: PLC0415

            cap = MAX_INTERACTIVE_FETCHES_PER_SESSION
        self._cap: int = cap
        self._count: int = 0

    @property
    def count(self) -> int:
        """Number of allow() calls made so far this session."""
        return self._count

    @property
    def cap(self) -> int:
        """The per-session ceiling configured at construction."""
        return self._cap

    def allow(self) -> bool:
        """Attempt one interactive fetch.

        Returns True when count < cap (fetch allowed); False at the inclusive
        breach (count >= cap, fetch blocked).  The internal counter increments
        on every call regardless of whether the fetch proceeds.
        """
        self._count += 1
        if self._count > self._cap:
            _logger.warning(
                "InteractiveRateLimiter: cap exceeded count=%d cap=%d",
                self._count,
                self._cap,
            )
            return False
        return True

    def mark_unavailable(
        self,
        brief: InvestmentBrief,  # type: ignore[name-defined]  # noqa: F821
        section: str,
    ) -> None:
        """Convenience wrapper: mark section data-unavailable on the brief.

        Delegates to the module-level mark_section_unavailable helper so the
        caller does not need to import it separately.
        """
        mark_section_unavailable(brief, section)
