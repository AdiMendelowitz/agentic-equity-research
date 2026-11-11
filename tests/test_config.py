"""Tests for config: Settings validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError


class TestSettingsDefaults:
    def test_cache_db_path_default(self, monkeypatch) -> None:
        from aer.config import Settings

        monkeypatch.delenv("CACHE_DB_PATH", raising=False)
        p = Path(Settings().cache_db_path)
        assert p.is_absolute()
        assert p.name == "aer.db"

        p = Path(Settings().cache_db_path)
        assert p.is_absolute()
        assert p.name == "aer.db"
        assert p.parent.name == ".cache"

    def test_cache_db_path_relative_env_resolved(self) -> None:
        from aer.config import Settings

        s = Settings(cache_db_path=".cache/aer.db")
        assert Path(s.cache_db_path).is_absolute()

    def test_port_default(self) -> None:
        from aer.config import Settings

        assert Settings().port == 8501

    def test_extra_fields_ignored(self) -> None:
        from aer.config import Settings

        s = Settings(unknown_field_xyz="value")  # type: ignore[call-arg]
        assert not hasattr(s, "unknown_field_xyz")


class TestSettingsPortValidation:
    def test_port_min_valid(self) -> None:
        from aer.config import Settings

        assert Settings(port=1).port == 1

    def test_port_max_valid(self) -> None:
        from aer.config import Settings

        assert Settings(port=65535).port == 65535

    def test_port_zero_raises(self) -> None:
        from aer.config import Settings

        with pytest.raises(ValidationError):
            Settings(port=0)

    def test_port_above_max_raises(self) -> None:
        from aer.config import Settings

        with pytest.raises(ValidationError):
            Settings(port=65536)
