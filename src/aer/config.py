"""Settings for agentic-equity-research. All LLM routing lives in src/aer/utils/llm_client.py."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aer.paths import ENV_FILE as _ENV_FILE
from aer.paths import PROJECT_ROOT as _PROJECT_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    # Data APIs
    fred_api_key: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "agentic-equity-research/0.1"
    edgar_identity: str = ""  # Must be set to "App Name name@example.com" per SEC EDGAR fair-use policy
    # Social Intelligence: Optional sources
    twitter_bearer_token: str = ""  # X API v2 bearer token (enables Twitter extraction when set + llm_provider=twitter)
    discord_bot_token: str = ""  # Discord bot token for message history fetching

    # Cache - relative paths (from .env or default) are resolved against project root, not CWD
    cache_db_path: str = str(_PROJECT_ROOT / ".cache" / "aer.db")

    @field_validator("cache_db_path", mode="after")
    @classmethod
    def _resolve_db_path(cls, v: str) -> str:
        p = Path(v)
        return str(p if p.is_absolute() else _PROJECT_ROOT / p)

    # App
    port: Annotated[int, Field(ge=1, le=65535)] = 8501

    # Feature flags
    news_agent_enabled: bool = False
    lseg_enabled: bool = False
    # Alt-data agents (backfill) - default OFF until a precision baseline + eval exist.
    agent_enabled_instagram: bool = False
    agent_enabled_layoffs: bool = False
    # Tier-1 data-pull audit logging (transient firehose). OFF by default; enable to capture
    # per-fetch row/date/NaN/cache diagnostics while hunting data-correctness bugs.
    data_pull_audit_enabled: bool = False
    # Social Intelligence: Optional feature flags
    llm_provider: str = "gemini"  # "gemini" (default), "claude", or "twitter" (enables X API extraction)
    social_synthesis_enabled: bool = False  # Layer 2 LLM synthesis for social consensus
    risk_free_rate: float = Field(default=0.045)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Clear the lru_cache on get_settings(). For use in tests that need fresh Settings."""
    get_settings.cache_clear()
