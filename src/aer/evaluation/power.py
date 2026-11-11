"""Sample-size and detectable-effect arithmetic for a one-sample test of a proportion (normal approximation)."""

import math
from statistics import NormalDist


def _check_probability(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value}")


def required_n(p0: float, p1: float, alpha: float = 0.05, power: float = 0.80, two_sided: bool = True) -> int:
    for name, value in (("p0", p0), ("p1", p1), ("alpha", alpha), ("power", power)):
        _check_probability(name, value)
    if p0 == p1:
        raise ValueError("p0 and p1 must differ")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2) if two_sided else normal.inv_cdf(1 - alpha)
    z_beta = normal.inv_cdf(power)
    numerator = z_alpha * math.sqrt(p0 * (1 - p0)) + z_beta * math.sqrt(p1 * (1 - p1))
    return math.ceil((numerator / (p1 - p0)) ** 2)


def detectable_effect(n: int, p0: float = 0.5, alpha: float = 0.05, power: float = 0.80) -> float:
    upper = 0.999
    if required_n(p0, upper, alpha, power) > n:
        raise ValueError(f"n={n} is too small to detect any p1 below {upper} against p0={p0}")
    lower = p0
    while upper - lower > 1e-4:
        middle = (lower + upper) / 2
        if required_n(p0, middle, alpha, power) <= n:
            upper = middle
        else:
            lower = middle
    return upper
