import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.model_selection import TimeSeriesSplit

from aer.evaluation.power import required_n
from aer.evaluation.spec import is_correct
from aer.ml.constants import CV_TEST_SIZE_BARS, MIN_TRAINING_BARS, N_CV_SPLITS, PURGE_GAP_BARS

ROOT = Path(__file__).resolve().parent.parent
PROFILE = settings(max_examples=100, deadline=None)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_figures", ROOT / "docs" / "report" / "build_figures.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_figures = _load()


@PROFILE
@given(
    p0=st.floats(min_value=0.3, max_value=0.7),
    effects=st.tuples(st.floats(min_value=0.02, max_value=0.2), st.floats(min_value=0.02, max_value=0.2)),
)
def test_required_n_non_increasing_in_effect(p0: float, effects: tuple[float, float]) -> None:
    small, large = sorted(effects)
    assert required_n(p0, p0 + large) <= required_n(p0, p0 + small)


@PROFILE
@given(data=st.data(), n=st.integers(min_value=1, max_value=5000))
def test_wilson_contains_point_estimate_and_narrows(data: st.DataObject, n: int) -> None:
    k = data.draw(st.integers(min_value=0, max_value=n))
    lo, hi = build_figures.wilson(k, n)
    tolerance = 1e-12
    assert 0.0 <= lo <= k / n + tolerance
    assert k / n - tolerance <= hi <= 1.0
    lo4, hi4 = build_figures.wilson(4 * k, 4 * n)
    assert hi4 - lo4 <= hi - lo + tolerance


@PROFILE
@given(n=st.integers(min_value=MIN_TRAINING_BARS, max_value=2000))
def test_purge_gap_holds_for_every_fold(n: int) -> None:
    splitter = TimeSeriesSplit(n_splits=N_CV_SPLITS, gap=PURGE_GAP_BARS, test_size=CV_TEST_SIZE_BARS)
    for train, test in splitter.split(np.arange(n)):
        assert test.min() - train.max() - 1 >= PURGE_GAP_BARS


@PROFILE
@given(
    forward_return=st.floats(min_value=-0.2, max_value=0.2, allow_nan=False),
    cost=st.floats(min_value=1e-6, max_value=0.049, allow_nan=False),
)
def test_is_correct_classes_are_exclusive(forward_return: float, cost: float) -> None:
    assert not (is_correct("BUY", forward_return, cost) and is_correct("SELL", forward_return, cost))
    on_boundary = forward_return in (cost, -cost)
    if not on_boundary:
        correct = [is_correct(name, forward_return, cost) for name in ("BUY", "SELL", "HOLD")]
        assert sum(correct) == 1
