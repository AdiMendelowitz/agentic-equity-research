"""class guard: a constant name defined in both constants.py and cost_thresholds.py must carry
the same value there. Two concepts need two names (e.g. GATE_G5_OLLAMA_UPTIME_FLOOR vs
KILL_SWITCH_OLLAMA_UPTIME_FLOOR)."""

from __future__ import annotations

import aer.utils.constants as constants
import aer.utils.cost_thresholds as cost_thresholds


def _public_upper(module) -> dict[str, object]:
    return {k: v for k, v in vars(module).items() if k.isupper() and not k.startswith("_")}


def test_shared_constant_names_agree() -> None:
    a, b = _public_upper(constants), _public_upper(cost_thresholds)
    conflicts = {k: (a[k], b[k]) for k in a.keys() & b.keys() if a[k] != b[k]}
    assert conflicts == {}, f"same name, different value: {conflicts}"


def test_gate_g5_and_cost_floor_are_distinct_names() -> None:
    assert constants.GATE_G5_OLLAMA_UPTIME_FLOOR == 0.95
    assert cost_thresholds.KILL_SWITCH_OLLAMA_UPTIME_FLOOR == 0.80
    assert not hasattr(constants, "KILL_SWITCH_OLLAMA_UPTIME_FLOOR")
