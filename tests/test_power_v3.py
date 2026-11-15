from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("power_v3", ROOT / "docs" / "report" / "power_v3.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["power_v3"] = module
    spec.loader.exec_module(module)
    return module


power_v3 = _load()


def test_sd_upper_hand_value() -> None:
    # the chi-square 0.20 quantile with 19 degrees of freedom is 13.716, so the factor is sqrt(19 / 13.716) = 1.1770
    assert power_v3.sd_upper(1.0, 20, 0.8) == pytest.approx(1.1770, abs=1e-3)


def test_sd_upper_scales_with_sd_and_exceeds_it() -> None:
    assert power_v3.sd_upper(0.05, 20, 0.8) == pytest.approx(0.05 * power_v3.sd_upper(1.0, 20, 0.8))
    assert power_v3.sd_upper(0.05, 20, 0.8) > 0.05


def test_superiority_power_equals_alpha_without_effect() -> None:
    assert power_v3.power_superiority(30, 0.0, 0.05, 0.05) == pytest.approx(0.05, abs=1e-9)


def test_power_increases_with_n() -> None:
    for power_fn in (power_v3.power_superiority, power_v3.power_tost):
        assert power_fn(20, 0.03, 0.05, 0.05) < power_fn(40, 0.03, 0.05, 0.05)


def test_power_increases_with_delta() -> None:
    assert power_v3.power_superiority(30, 0.01, 0.05, 0.05) < power_v3.power_superiority(30, 0.03, 0.05, 0.05)
    assert power_v3.power_tost(30, 0.02, 0.05, 0.05) < power_v3.power_tost(30, 0.04, 0.05, 0.05)


def test_required_seeds_is_monotone_in_delta() -> None:
    small = power_v3.required_seeds(power_v3.power_superiority, 0.01, 0.05, 0.05, 0.8, 20, maximum=2000)
    large = power_v3.required_seeds(power_v3.power_superiority, 0.02, 0.05, 0.05, 0.8, 20, maximum=2000)
    assert small > large


def test_required_seeds_respects_minimum() -> None:
    assert power_v3.required_seeds(power_v3.power_superiority, 0.5, 0.05, 0.05, 0.8, 20) == 20
    assert power_v3.required_seeds(power_v3.power_superiority, 0.5, 0.05, 0.05, 0.8, 35) == 35


def test_required_seeds_raises_when_unreachable() -> None:
    with pytest.raises(ValueError):
        power_v3.required_seeds(power_v3.power_superiority, 0.001, 0.05, 0.05, 0.8, 20, maximum=100)


def test_required_seeds_close_to_normal_approximation() -> None:
    delta, sd, alpha, target = 0.005, 0.05, 0.05, 0.8
    approx = ((norm.ppf(1 - alpha) + norm.ppf(target)) * sd / delta) ** 2
    exact = power_v3.required_seeds(power_v3.power_superiority, delta, sd, alpha, target, 20, maximum=2000)
    assert abs(exact - approx) <= 3


def test_power_tost_tiny_delta_is_near_zero_and_large_delta_near_one() -> None:
    assert power_v3.power_tost(20, 1e-4, 0.05, 0.05) < 0.01
    assert power_v3.power_tost(100, 0.1, 0.05, 0.05) > 0.99
