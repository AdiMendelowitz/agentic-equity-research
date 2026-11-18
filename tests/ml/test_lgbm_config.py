"""LightGBM bagging is off and the configuration says so, with subsample 1.0 and no subsample frequency."""

import numpy as np

from aer.ml.constants import LGBM_SUBSAMPLE
from aer.ml.models import _make_lgbm


def test_lgbm_row_subsampling_is_off_and_stated() -> None:
    params = _make_lgbm().get_params()
    assert LGBM_SUBSAMPLE == 1.0
    assert params["subsample"] == 1.0
    assert params["subsample_freq"] == 0
    assert not {"bagging_freq", "bagging_fraction"} & params.keys()


def test_subsample_without_frequency_changes_no_fit() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 6))
    y = (x[:, 0] + rng.normal(size=300) > 0).astype(int)
    stated = _make_lgbm().fit(x, y)
    inert = _make_lgbm().set_params(subsample=0.5).fit(x, y)
    assert stated.booster_.dump_model()["tree_info"] == inert.booster_.dump_model()["tree_info"]
    np.testing.assert_array_equal(stated.predict_proba(x), inert.predict_proba(x))
