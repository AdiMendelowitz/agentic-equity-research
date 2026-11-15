"""Specification constants for the pre-registered detection control, version 3.

This module is the single source of every number the version 3 analysis uses. Changing any value after the hash of
the specification (docs/prereg/control-v3.md and this file) has been published invalidates the pre-registration.
"""

from __future__ import annotations

SPEC_ID: str = "control-v3"
SIGNALS: tuple[float, ...] = (0.0, 0.3, 0.6)
DELTA: float = 0.03
ALPHA: float = 0.05
POWER: float = 0.80
SD_UPPER_CONFIDENCE: float = 0.80
MIN_SEEDS: int = 20
SEED_OFFSET: int = 1000
N_SEEDS: int | None = 46


def seeds() -> tuple[int, ...]:
    if N_SEEDS is None:
        raise ValueError("N_SEEDS is not set: run docs/report/power_v3.py and record its output before publication")
    return tuple(range(SEED_OFFSET, SEED_OFFSET + N_SEEDS))
