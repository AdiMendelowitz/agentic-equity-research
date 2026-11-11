import pytest

from aer.evaluation.power import detectable_effect, required_n


def test_required_n_known_value() -> None:
    # z_a = 1.959964, z_b = 0.841621, sqrt(0.5 * 0.5) = 0.5, sqrt(0.55 * 0.45) = sqrt(0.2475) = 0.497494
    # numerator = 0.979982 + 0.841621 * 0.497494 = 0.979982 + 0.418701 = 1.398683
    # (1.398683 / 0.05) ** 2 = 27.97366 ** 2 = 782.53, rounded up to 783
    assert required_n(0.50, 0.55) == 783


def test_required_n_roughly_symmetric() -> None:
    up = required_n(0.50, 0.55)
    down = required_n(0.50, 0.45)
    assert abs(up - down) / up < 0.01


def test_smaller_effect_needs_larger_n() -> None:
    assert required_n(0.50, 0.53) > required_n(0.50, 0.55) > required_n(0.50, 0.60)


def test_one_sided_needs_fewer_than_two_sided() -> None:
    assert required_n(0.50, 0.55, two_sided=False) < required_n(0.50, 0.55, two_sided=True)


@pytest.mark.parametrize(
    ("p0", "p1"),
    [(0.5, 0.5), (0.0, 0.5), (0.5, 1.0), (-0.1, 0.5), (0.5, 1.2)],
)
def test_required_n_invalid_probabilities(p0: float, p1: float) -> None:
    with pytest.raises(ValueError):
        required_n(p0, p1)


def test_required_n_invalid_alpha_and_power() -> None:
    with pytest.raises(ValueError):
        required_n(0.5, 0.55, alpha=0.0)
    with pytest.raises(ValueError):
        required_n(0.5, 0.55, power=1.0)


def test_detectable_effect_brackets() -> None:
    # required_n(0.50, 0.55) = 783 > 707, so the answer exceeds 0.55.
    # p1 = 0.56: sqrt(0.56 * 0.44) = sqrt(0.2464) = 0.496387; numerator = 0.979982 + 0.841621 * 0.496387 = 1.397751;
    # (1.397751 / 0.06) ** 2 = 23.29585 ** 2 = 542.7, rounded up to 543 <= 707, so the answer is below 0.56.
    assert required_n(0.50, 0.55) > 707
    assert required_n(0.50, 0.56) <= 707
    assert 0.55 < detectable_effect(707) < 0.56


def test_detectable_effect_is_consistent_with_required_n() -> None:
    effect = detectable_effect(1000)
    assert required_n(0.5, effect) <= 1000
    assert required_n(0.5, effect - 2e-4) > 1000


def test_detectable_effect_too_small_sample() -> None:
    with pytest.raises(ValueError):
        detectable_effect(3)
