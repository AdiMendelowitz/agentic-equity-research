"""Net-liquidity calendar-alignment tests.

Covers the named trend threshold constant (boundary semantics) and the
`_align_net_liquidity` helper: discriminating `.asof()` alignment (not positional
`.iloc[-1]`), a revived non-None trend on realistic weekly+daily data, the
insufficient-history NaN edge, an economic-plausibility band, sort-guarding, and
the latest-value NaN guard.
"""

import pandas as pd
import pytest

from aer.data.macro import _align_net_liquidity
from aer.utils.constants import (
    NET_LIQUIDITY_DAILY_LOOKBACK_PERIODS,
    NET_LIQUIDITY_TREND_THRESHOLD_BN,
)


def _classify(delta: float) -> str:
    """Mirror the strict-inequality trend classification used in _align_net_liquidity."""
    if delta > NET_LIQUIDITY_TREND_THRESHOLD_BN:
        return "expanding"
    if delta < -NET_LIQUIDITY_TREND_THRESHOLD_BN:
        return "contracting"
    return "neutral"


class TestNetLiquidityTrendThresholdConstant:
    def test_threshold_value(self) -> None:
        assert NET_LIQUIDITY_TREND_THRESHOLD_BN == 50.0

    def test_daily_lookback_reaches_prior_week(self) -> None:
        # Must exceed one trading week (5 business days) so .asof(prev_anchor) resolves.
        assert NET_LIQUIDITY_DAILY_LOOKBACK_PERIODS > 5

    def test_delta_equal_to_threshold_is_neutral(self) -> None:
        # Strict `>` / `<`: a delta exactly on the threshold is neutral, not expanding.
        assert _classify(NET_LIQUIDITY_TREND_THRESHOLD_BN) == "neutral"
        assert _classify(-NET_LIQUIDITY_TREND_THRESHOLD_BN) == "neutral"

    def test_delta_one_above_threshold_is_directional(self) -> None:
        assert _classify(NET_LIQUIDITY_TREND_THRESHOLD_BN + 1.0) == "expanding"
        assert _classify(-NET_LIQUIDITY_TREND_THRESHOLD_BN - 1.0) == "contracting"


def _series(pairs: list[tuple[str, float]]) -> pd.Series:
    """Build a date-indexed Series from (iso_date, value) pairs."""
    dates, values = zip(*pairs)
    return pd.Series(list(values), index=pd.to_datetime(list(dates)))


class TestData04NetLiquidityAlignment:
    def test_asof_alignment_not_positional(self) -> None:
        """The daily RRP series carries an observation AFTER the
        WALCL anchor, so positional .iloc[-1] would pick the post-anchor value (540)
        while .asof(anchor) picks the on/before value (503). Asserting 503 fails
        under the old positional implementation - this is what pins the fix."""
        walcl_bn = _series([("2024-01-10", 7000.0), ("2024-01-17", 7010.0), ("2024-01-24", 7020.0)])
        # 2024-01-25 and 2024-01-29 postdate the 2024-01-24 anchor; iloc[-1] would be 540.0.
        rrp = _series(
            [
                ("2024-01-22", 501.0),
                ("2024-01-23", 502.0),
                ("2024-01-24", 503.0),
                ("2024-01-25", 510.0),
                ("2024-01-29", 540.0),
            ]
        )
        tga = _series([("2024-01-17", 700.0), ("2024-01-24", 700.0)])

        result = _align_net_liquidity(walcl_bn, rrp, tga)

        assert result["rrp_bn"] == pytest.approx(503.0)
        assert result["rrp_bn"] != pytest.approx(540.0)  # the positional .iloc[-1] value
        assert result["net_liquidity_bn"] == pytest.approx(7020.0 - 503.0 - 700.0)

    def test_positive_trend_revived(self) -> None:
        """With the daily RRP series spanning >=2 weeks (as the widened
        production lookback guarantees), .asof(prev_anchor) resolves and the trend is a
        real non-None label rather than perpetually None."""
        walcl_bn = _series([("2024-01-17", 7000.0), ("2024-01-24", 7100.0)])
        rrp = _series(
            [
                ("2024-01-12", 500.0),
                ("2024-01-16", 500.0),
                ("2024-01-17", 500.0),
                ("2024-01-23", 500.0),
                ("2024-01-24", 500.0),
            ]
        )
        tga = _series([("2024-01-17", 700.0), ("2024-01-24", 700.0)])

        result = _align_net_liquidity(walcl_bn, rrp, tga)

        # net_latest=5900, net_prev=5800, delta=100 > 50 -> expanding
        assert result["net_liquidity_trend"] == "expanding"

    def test_trend_none_on_insufficient_daily_history(self) -> None:
        """Reframed edge: when the daily series genuinely has no observation reaching
        prev_anchor, the trend falls back to None without raising (true insufficient
        history, NOT the normal case the widened lookback fixes)."""
        walcl_bn = _series([("2024-01-17", 7000.0), ("2024-01-24", 7100.0)])
        # All daily RRP obs postdate prev_anchor (2024-01-17).
        rrp = _series([("2024-01-22", 500.0), ("2024-01-23", 500.0), ("2024-01-24", 500.0)])
        tga = _series([("2024-01-17", 700.0), ("2024-01-24", 700.0)])

        result = _align_net_liquidity(walcl_bn, rrp, tga)

        assert result["net_liquidity_trend"] is None
        assert result["net_liquidity_bn"] is not None  # latest path still resolves

    def test_economic_plausibility_band(self) -> None:
        """Realistic magnitudes (WALCL ~7000bn, RRP ~500bn,
        TGA ~700bn) net liquidity must land in a sane billions band. A dropped or
        doubled millions->billions conversion would push it orders of magnitude out."""
        walcl_bn = _series([("2024-01-24", 7000.0)])
        rrp = _series([("2024-01-24", 500.0)])
        tga = _series([("2024-01-24", 700.0)])

        result = _align_net_liquidity(walcl_bn, rrp, tga)

        assert isinstance(result["net_liquidity_bn"], float)
        assert 4000.0 < result["net_liquidity_bn"] < 7000.0

    def test_sort_guard_unsorted_index(self) -> None:
        """A deliberately reversed index must not raise (helper sort_index guards asof)."""
        walcl_bn = _series([("2024-01-24", 7100.0), ("2024-01-17", 7000.0)])  # reversed
        rrp = _series([("2024-01-24", 500.0), ("2024-01-17", 500.0)])
        tga = _series([("2024-01-24", 700.0), ("2024-01-17", 700.0)])

        result = _align_net_liquidity(walcl_bn, rrp, tga)

        assert isinstance(result, dict)
        assert result["net_liquidity_bn"] is not None

    def test_latest_nan_guard_all_none(self) -> None:
        """When the anchor predates all RRP observations, every output value is None."""
        walcl_bn = _series([("2024-01-01", 7000.0)])
        rrp = _series([("2024-01-22", 500.0)])  # entirely after the anchor
        tga = _series([("2024-01-22", 700.0)])

        result = _align_net_liquidity(walcl_bn, rrp, tga)

        assert all(v is None for v in result.values())
