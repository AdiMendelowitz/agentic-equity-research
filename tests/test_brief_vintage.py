"""Tests for per-field vintage timestamp tracking in InvestmentBrief."""

import time

from aer.models import InvestmentBrief


class TestVintageTracking:
    """Test suite for brief.vintage dict population and lifecycle."""

    def test_brief_vintage_populated_after_build(self):
        """Verify that all 8 vintage keys are populated after build_brief() completes."""
        # Create a mock brief with vintage dict
        now = time.time()
        brief = InvestmentBrief(
            ticker="AAPL",
            company_name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology",
            recommendation="BUY",
            conviction=0.85,
            summary="Test brief",
            vintage={
                "prices": now - 300,  # 5 minutes ago
                "consensus": now - 600,  # 10 minutes ago
                "macro": now - 1200,  # 20 minutes ago
                "earnings": now - 1800,  # 30 minutes ago
                "ownership": now - 2400,  # 40 minutes ago
                "options": now - 3000,  # 50 minutes ago
                "sentiment": now - 3600,  # 60 minutes ago
                "factors": now - 7200,  # 2 hours ago
            },
        )

        # Verify all 8 keys present
        expected_keys = {"prices", "consensus", "macro", "earnings", "ownership", "options", "sentiment", "factors"}
        assert set(brief.vintage.keys()) == expected_keys, f"Missing keys: {expected_keys - set(brief.vintage.keys())}"

    def test_vintage_timestamps_recent(self):
        """Verify that vintage timestamps are recent (less than 1 hour old)."""
        now = time.time()
        brief = InvestmentBrief(
            ticker="AAPL",
            company_name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology",
            recommendation="BUY",
            conviction=0.85,
            summary="Test brief",
            vintage={
                "prices": now - 300,  # 5 minutes ago
                "consensus": now - 600,  # 10 minutes ago
                "macro": now - 1200,  # 20 minutes ago
                "earnings": now - 1800,  # 30 minutes ago
                "ownership": now - 2400,  # 40 minutes ago
                "options": now - 3000,  # 50 minutes ago
                "sentiment": now - 3600,  # 60 minutes ago
                "factors": now - 7200,  # 2 hours ago (older)
            },
        )

        # All timestamps should be Unix floats
        for key, ts in brief.vintage.items():
            assert isinstance(ts, (int, float)), f"vintage[{key}] is not numeric: {type(ts)}"
            # Verify it's a reasonable timestamp (not in the future, not ancient)
            assert ts <= now, f"vintage[{key}]={ts} is in the future (now={now})"
            assert ts > now - 86400, f"vintage[{key}]={ts} is more than 24h old"

    def test_vintage_dict_serialization(self):
        """Verify that vintage dict survives JSON serialization (for cache store/retrieve)."""
        now = time.time()
        brief = InvestmentBrief(
            ticker="AAPL",
            company_name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology",
            recommendation="BUY",
            conviction=0.85,
            summary="Test brief",
            vintage={
                "prices": now - 300,
                "consensus": now - 600,
                "macro": now - 1200,
                "earnings": now - 1800,
                "ownership": now - 2400,
                "options": now - 3000,
                "sentiment": now - 3600,
                "factors": now - 7200,
            },
        )

        # Serialize to dict (JSON-safe)
        serialized = brief.model_dump(mode="json")
        assert "vintage" in serialized, "vintage dict missing from serialized brief"

        # Verify vintage structure after serialization
        vintage_serialized = serialized["vintage"]
        assert isinstance(vintage_serialized, dict), f"vintage not dict after serialization: {type(vintage_serialized)}"
        assert len(vintage_serialized) == 8, f"vintage has {len(vintage_serialized)} keys, expected 8"

        # Deserialize back to brief
        brief_restored = InvestmentBrief.model_validate(serialized)
        assert brief_restored.vintage == brief.vintage, "vintage dict changed after serialize/deserialize"

    def test_backward_compat_old_brief_without_vintage(self):
        """Verify that old briefs without vintage dict don't crash when accessed."""
        # Create brief with empty vintage (backward compatible)
        brief = InvestmentBrief(
            ticker="AAPL",
            company_name="Apple Inc.",
            exchange="NASDAQ",
            sector="Technology",
            recommendation="BUY",
            conviction=0.85,
            summary="Test brief",
            vintage={},  # Empty, as if from old brief
        )

        # Should not crash; vintage should be empty dict
        assert brief.vintage == {}, f"Expected empty vintage, got {brief.vintage}"
        assert isinstance(brief.vintage, dict), f"vintage is not dict: {type(brief.vintage)}"

        # Accessing missing key should return None (or raise KeyError — define behavior)
        assert brief.vintage.get("prices") is None, "old brief should have None for missing vintage key"


class TestVintageTrackerSession:
    """Test suite for vintage_tracker session management."""

    def test_session_start_and_retrieve(self):
        """Verify that starting a session allows recording and retrieving timestamps."""
        from aer.utils.vintage_tracker import clear_session, get_session_vintage, record_vintage, start_session

        # Start a new session
        start_session()

        # Record some timestamps
        now = time.time()
        record_vintage("prices", now - 300)
        record_vintage("consensus", now - 600)
        record_vintage("macro", now - 1200)

        # Retrieve the session
        vintage = get_session_vintage()

        # Verify all recorded sources are present
        assert "prices" in vintage, "prices not in session vintage"
        assert "consensus" in vintage, "consensus not in session vintage"
        assert "macro" in vintage, "macro not in session vintage"

        # Verify timestamps match
        assert abs(vintage["prices"] - (now - 300)) < 0.1, "prices timestamp mismatch"
        assert abs(vintage["consensus"] - (now - 600)) < 0.1, "consensus timestamp mismatch"
        assert abs(vintage["macro"] - (now - 1200)) < 0.1, "macro timestamp mismatch"

        # Unrecorded keys should not be present
        assert "earnings" not in vintage, "earnings should not be recorded"

        # Clean up
        clear_session()

    def test_session_clear_resets_tracking(self):
        """Verify that clear_session removes all tracking data."""
        from aer.utils.vintage_tracker import clear_session, get_session_vintage, record_vintage, start_session

        start_session()
        now = time.time()
        record_vintage("prices", now)

        # Verify data is there
        vintage_before = get_session_vintage()
        assert "prices" in vintage_before, "prices should be recorded"

        # Clear and verify it's gone
        clear_session()
        vintage_after = get_session_vintage()
        assert "prices" not in vintage_after, "prices should be cleared"
        assert len(vintage_after) == 0, "session should be empty after clear"

    def test_session_records_current_time_if_none_provided(self):
        """Verify that record_vintage uses current time if timestamp is None."""
        from aer.utils.vintage_tracker import clear_session, get_session_vintage, record_vintage, start_session

        start_session()
        before = time.time()
        record_vintage("sentiment")  # No timestamp provided
        after = time.time()

        vintage = get_session_vintage()
        assert "sentiment" in vintage, "sentiment should be recorded"
        ts = vintage["sentiment"]

        # Timestamp should be between before and after calls
        assert before <= ts <= after, f"timestamp {ts} not in range [{before}, {after}]"

        clear_session()
