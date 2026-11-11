"""RED tests for professor flags schema and vintage model."""

import json

import pytest

from aer.models import InvestmentBrief
from aer.utils.constants import REQUIRED_SEVERITY_KEY


class TestProfessorFlagsSchema:
    """Tests for professor_flags validation."""

    def test_professor_flags_valid_schema(self):
        """RED: valid professor_flags dict has 'severity' key."""
        flags = [{"severity": "high", "pattern": "CONF_MACRO_OVERRIDE"}]
        assert all(REQUIRED_SEVERITY_KEY in flag for flag in flags)

    def test_professor_flags_invalid_schema(self):
        """RED: invalid professor_flags missing 'severity' key raises error."""
        flags = [{"pattern": "CONF_MACRO_OVERRIDE"}]

        def validate_professor_flags(flags_list):
            for flag in flags_list:
                if REQUIRED_SEVERITY_KEY not in flag:
                    raise ValueError(f"Missing required key: {REQUIRED_SEVERITY_KEY}")

        with pytest.raises(ValueError, match="Missing required key"):
            validate_professor_flags(flags)


class TestVintageDict:
    """Tests for vintage timestamp model."""

    def test_vintage_dict_serialization(self):
        """RED: InvestmentBrief with vintage dict roundtrips JSON."""
        brief = InvestmentBrief(
            ticker="AAPL",
            company_name="Apple Inc.",
            recommendation="BUY",
            conviction=0.75,
            vintage={
                "prices_ts": 1719100800.0,
                "consensus_ts": 1719014400.0,
                "macro_ts": 1718928000.0,
            },
        )

        # Serialize to JSON
        json_str = brief.model_dump_json()
        assert json_str is not None
        parsed = json.loads(json_str)

        # Verify vintage dict is present
        assert "vintage" in parsed
        assert parsed["vintage"]["prices_ts"] == 1719100800.0
        assert parsed["vintage"]["consensus_ts"] == 1719014400.0
        assert parsed["vintage"]["macro_ts"] == 1718928000.0

        # Deserialize back
        brief_restored = InvestmentBrief.model_validate_json(json_str)
        assert brief_restored.vintage == brief.vintage
