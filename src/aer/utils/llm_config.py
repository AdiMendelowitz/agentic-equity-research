"""
LLM routing configuration — single source of truth for all model assignments.

All values are read from environment variables at module import time.
To override in tests: monkeypatch env vars and reload this module with
importlib.reload(aer.utils.llm_config).

Provider strings use Literal typing — valid values are:
  "openrouter", "nvidia_nim", "groq", "ollama", "anthropic"

Environment variables:
  Required:
    ANTHROPIC_API_KEY               Anthropic SDK (fallback)
    OPENROUTER_API_KEY              OpenRouter free tier (primary free)
  Optional:
    NVIDIA_NIM_API_KEY              NVIDIA NIM: EXTRACTION secondary; ANALYSIS and SYNTHESIS primary (R-A)
    GROQ_API_KEY                    Groq free plan: ANALYSIS and SYNTHESIS secondary (R-A)

  Routing (leave blank to use defaults listed below):
    MODEL_EXTRACTION_PRIMARY        default: openrouter/free
    MODEL_EXTRACTION_SECONDARY      default: nvidia/nemotron-3-super-120b-a12b on NIM
                                    (deepseek/deepseek-chat-v3.1:free without a NIM key)
    MODEL_ANALYSIS_PRIMARY          default: nvidia/nemotron-3-super-120b-a12b on NIM
                                    (nvidia/nemotron-3-super-120b-a12b:free on OpenRouter without a NIM key)
    MODEL_ANALYSIS_SECONDARY        default: openai/gpt-oss-20b on Groq
                                    (google/gemma-4-31b-it:free on OpenRouter without a Groq key)
    MODEL_SYNTHESIS_PRIMARY         default: as MODEL_ANALYSIS_PRIMARY
    MODEL_SYNTHESIS_SECONDARY       default: openai/gpt-oss-120b on Groq
                                    (nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free without a Groq key)
    MODEL_EXTRACTION_FALLBACK       default: claude-haiku-4-5-20251001
    MODEL_ANALYSIS_FALLBACK         default: claude-haiku-4-5-20251001
    MODEL_SYNTHESIS_FALLBACK        default: claude-sonnet-4-6

  Behaviour:
    FREE_MODEL_CONTEXT_LIMIT        default: 32000  (tokens; skip free if exceeded)
    FREE_MODEL_TIMEOUT_SECONDS      default: 15.0
    FREE_MODEL_MAX_RETRIES          default: 2      (per provider, for 429 only)
"""

import dataclasses
import os
from typing import Final, Literal

ProviderName = Literal["openrouter", "nvidia_nim", "groq", "ollama", "anthropic"]

# meta/llama-3.3-70b-instruct reached end of life on NIM on 2026-08-26 (HTTP 410). Chosen from the live
# catalogue by scripts/dev/nim_smoke.py (nim_smoke_20260928-101608: valid JSON 2 of 2, 5.4-9.4 s); a small sample.
_NIM_CHAT_MODEL: Final[str] = "nvidia/nemotron-3-super-120b-a12b"

# plan_llm_route_2026-09-28 item 1: nemotron on NIM answered in 5-9 s on a short prompt (nim_smoke) but hit the
# 15 s free timeout on 8 of 18 pipeline calls (profile_20260928-103322); NIM gets its own ceiling.
NIM_TIMEOUT_SECONDS: Final[float] = float(os.environ.get("NIM_TIMEOUT_SECONDS", "60.0"))

FREE_MODEL_CONTEXT_LIMIT: Final[int] = int(os.environ.get("FREE_MODEL_CONTEXT_LIMIT", "32000"))
FREE_MODEL_TIMEOUT: Final[float] = float(os.environ.get("FREE_MODEL_TIMEOUT_SECONDS", "15.0"))
FREE_MODEL_MAX_RETRIES: Final[int] = int(os.environ.get("FREE_MODEL_MAX_RETRIES", "2"))
ANTHROPIC_FALLBACK_TIMEOUT: Final[float] = float(os.environ.get("ANTHROPIC_FALLBACK_TIMEOUT_SECONDS", "30.0"))


@dataclasses.dataclass(frozen=True)
class TierConfig:
    primary_model: str
    primary_provider: ProviderName
    secondary_model: str
    secondary_provider: ProviderName
    fallback_model: str  # always routed to Anthropic


def load_tier_config() -> dict[str, TierConfig]:
    """
    Build tier routing config from environment variables.

    Called once at module import to populate TIER_CONFIG.
    Also callable directly in tests after monkeypatching env vars:
        importlib.reload(aer.utils.llm_config)
    """
    extraction_secondary_provider: ProviderName = "nvidia_nim" if os.environ.get("NVIDIA_NIM_API_KEY") else "openrouter"
    extraction_secondary_default = (
        _NIM_CHAT_MODEL if os.environ.get("NVIDIA_NIM_API_KEY") else "deepseek/deepseek-chat-v3.1:free"
    )

    # R-A (Adi 2026-09-28): ANALYSIS and SYNTHESIS go NVIDIA NIM first (no documented daily cap on the
    # free developer tier), Groq second (free plan, 1K requests/day), paid Anthropic last. Each free level
    # falls back to its previous OpenRouter default when its key is absent, so a machine without the keys
    # keeps the old route. OpenRouter free allows 50 requests/day without credit (profile 2026-09-28).
    has_nim = bool(os.environ.get("NVIDIA_NIM_API_KEY"))
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    free_primary_provider: ProviderName = "nvidia_nim" if has_nim else "openrouter"
    free_primary_default = _NIM_CHAT_MODEL if has_nim else "nvidia/nemotron-3-super-120b-a12b:free"
    free_secondary_provider: ProviderName = "groq" if has_groq else "openrouter"

    return {
        "EXTRACTION": TierConfig(
            primary_model=os.environ.get("MODEL_EXTRACTION_PRIMARY", "openrouter/free"),
            primary_provider="openrouter",
            secondary_model=os.environ.get("MODEL_EXTRACTION_SECONDARY", extraction_secondary_default),
            secondary_provider=extraction_secondary_provider,
            fallback_model=os.environ.get("MODEL_EXTRACTION_FALLBACK", "claude-haiku-4-5-20251001"),
        ),
        "ANALYSIS": TierConfig(
            primary_model=os.environ.get("MODEL_ANALYSIS_PRIMARY", free_primary_default),
            primary_provider=free_primary_provider,
            secondary_model=os.environ.get(
                "MODEL_ANALYSIS_SECONDARY", "openai/gpt-oss-20b" if has_groq else "google/gemma-4-31b-it:free"
            ),
            secondary_provider=free_secondary_provider,
            # DOWNGRADE: Haiku sufficient for per-agent narrative from structured numeric inputs
            fallback_model=os.environ.get("MODEL_ANALYSIS_FALLBACK", "claude-haiku-4-5-20251001"),
        ),
        "SYNTHESIS": TierConfig(
            primary_model=os.environ.get("MODEL_SYNTHESIS_PRIMARY", free_primary_default),
            primary_provider=free_primary_provider,
            secondary_model=os.environ.get(
                "MODEL_SYNTHESIS_SECONDARY",
                "openai/gpt-oss-120b" if has_groq else "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            ),
            secondary_provider=free_secondary_provider,
            fallback_model=os.environ.get("MODEL_SYNTHESIS_FALLBACK", "claude-sonnet-4-6"),
        ),
    }


TIER_CONFIG: Final[dict[str, TierConfig]] = load_tier_config()
