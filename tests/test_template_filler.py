"""Tests for src/aer/utils/template_filler.py - pure function tests, no I/O."""

from aer.utils.template_filler import fill_template


def test_fill_template_all_keys_present() -> None:
    """All placeholders resolved - filled equals expected string, missing_keys empty."""
    result = fill_template("{ticker} {view}", {"ticker": "AAPL", "view": "BUY"})
    assert result.filled == "AAPL BUY"
    assert result.missing_keys == []


def test_fill_template_structural_missing_returns_none() -> None:
    """Missing structural key → filled is None, missing key reported."""
    result = fill_template("{ticker} {view}", {"ticker": "AAPL"})
    assert result.filled is None
    assert "view" in result.missing_keys


def test_fill_template_missing_keys_always_populated() -> None:
    """When filled is None, missing_keys must be a non-empty list."""
    result = fill_template("{ticker} {view}", {"ticker": "AAPL"})
    assert result.filled is None
    assert isinstance(result.missing_keys, list)
    assert len(result.missing_keys) > 0


def test_fill_template_any_missing_key_blocks_fill() -> None:
    """Any missing key blocks fill - all placeholders are required."""
    result = fill_template("{ticker} {extra}", {"ticker": "AAPL"})
    assert result.filled is None
    assert "extra" in result.missing_keys


def test_fill_template_template_key_propagated() -> None:
    """template_key argument is stored on the result."""
    result = fill_template("{x}", {}, template_key="TMPL_TEST")
    assert result.template_key == "TMPL_TEST"


def test_fill_template_never_raises_on_empty_template() -> None:
    """Empty template string returns filled=='' without exception."""
    result = fill_template("", {})
    assert result.filled == ""


def test_fill_template_multiple_missing_keys() -> None:
    """All three missing placeholders are reported when context is empty."""
    result = fill_template("{a} {b} {c}", {})
    assert result.filled is None
    assert len(result.missing_keys) == 3
    assert set(result.missing_keys) == {"a", "b", "c"}


# --- Regime-tag support ---


def test_fill_template_regime_variant_selected() -> None:
    """regime_tag selects the suffixed template variant when available."""
    from unittest.mock import patch

    with patch("aer.utils.template_filler.get_template_text") as mock_gtt:
        mock_gtt.side_effect = lambda k: "risk variant {x}" if k == "TMPL_TEST_RISK" else None
        result = fill_template("{x}", {"x": "v"}, template_key="TMPL_TEST", regime_tag="risk")
    assert result.filled == "risk variant v"
    assert result.missing_keys == []


def test_fill_template_unknown_regime_fallback() -> None:
    """unknown regime_tag falls back to the base template without error."""
    result = fill_template("{ticker}", {"ticker": "AAPL"}, template_key="TMPL_TEST", regime_tag="nonexistent_tag")
    assert result.filled == "AAPL"
    assert result.missing_keys == []


def test_regime_to_template_tag_coverage() -> None:
    """REGIME_TO_TEMPLATE_TAG covers all 6 required regime labels."""
    from aer.utils.constants import REGIME_TO_TEMPLATE_TAG

    expected_labels = {"BULL_STEADY", "BULL_VOLATILE", "BEAR", "STAGFLATION", "RECOVERY", "UNKNOWN"}
    assert set(REGIME_TO_TEMPLATE_TAG.keys()) == expected_labels
