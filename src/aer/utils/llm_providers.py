"""
Provider adapters for OpenRouter, NVIDIA NIM and Groq.

Contract:
  - Each public function makes exactly ONE HTTP request.
  - Each public function returns a non-empty str or raises.
  - No retry logic here — retries are llm_client.py's responsibility. The OpenAI SDK retries twice by
    default, which tripled every timeout (nim_smoke: 60 s timeout, 181.7 s observed), so clients set max_retries=0.
  - No json_mode parameter on call_nvidia_nim — NIM does not reliably support
    response_format=json_object. Callers must append JSON instructions to
    user_message before calling call_nvidia_nim.
  - Provider clients are module-level singletons initialised lazily.
"""

import logging
import os
import threading
from typing import Any, Final

from aer.utils import usage_meter

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
NVIDIA_NIM_BASE_URL: Final[str] = "https://integrate.api.nvidia.com/v1"
GROQ_BASE_URL: Final[str] = "https://api.groq.com/openai/v1"  # console.groq.com/docs/openai
OPENROUTER_APP_REFERER: Final[str] = os.environ.get(
    "OPENROUTER_APP_REFERER", "https://github.com/market-intelligence-dashboard"
)
OPENROUTER_APP_TITLE: Final[str] = os.environ.get("OPENROUTER_APP_TITLE", "market-intelligence-dashboard")

_openrouter_client: Any = None  # openai.OpenAI, typed as Any to avoid import at module level
_nim_client: Any = None
_openrouter_lock: threading.Lock = threading.Lock()
_nim_lock: threading.Lock = threading.Lock()
_groq_client: Any = None
_groq_lock: threading.Lock = threading.Lock()


def _get_openrouter_client() -> Any:
    global _openrouter_client
    if _openrouter_client is None:
        with _openrouter_lock:
            if _openrouter_client is None:
                from openai import OpenAI  # deferred — avoids ImportError if openai not installed

                api_key = os.environ.get("OPENROUTER_API_KEY", "")
                if not api_key:
                    raise OSError("OPENROUTER_API_KEY not set — cannot use OpenRouter provider")
                _openrouter_client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, max_retries=0)
    return _openrouter_client


def _get_nim_client() -> Any:
    global _nim_client
    if _nim_client is None:
        with _nim_lock:
            if _nim_client is None:
                from openai import OpenAI

                api_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
                if not api_key:
                    raise OSError("NVIDIA_NIM_API_KEY not set — cannot use NVIDIA NIM provider")
                _nim_client = OpenAI(base_url=NVIDIA_NIM_BASE_URL, api_key=api_key, max_retries=0)
    return _nim_client


def _get_groq_client() -> Any:
    global _groq_client
    if _groq_client is None:
        with _groq_lock:
            if _groq_client is None:
                from openai import OpenAI

                api_key = os.environ.get("GROQ_API_KEY", "")
                if not api_key:
                    raise OSError("GROQ_API_KEY not set — cannot use Groq provider")
                _groq_client = OpenAI(base_url=GROQ_BASE_URL, api_key=api_key, max_retries=0)
    return _groq_client


def _build_messages(system_prompt: str, user_message: str) -> list[dict[str, str]]:
    """Build OpenAI-format messages list. Omits system message if empty."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    return messages


def call_openrouter(
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    json_mode: bool,
    timeout: float,
    temperature: float = 1.0,
) -> str:
    """
    Make one OpenRouter request. Returns non-empty response string or raises.

    json_mode: if True, sets response_format={"type": "json_object"}.
               The caller must also instruct the model to return JSON in the
               prompt — json_mode alone is not sufficient for all models.
    """
    client = _get_openrouter_client()
    messages = _build_messages(system_prompt, user_message)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
        "extra_headers": {
            "HTTP-Referer": OPENROUTER_APP_REFERER,
            "X-Title": OPENROUTER_APP_TITLE,
        },
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    usage_meter.record_openai_usage("openrouter", model, response)
    choices = response.choices
    if not choices:
        raise ValueError(f"Empty response from OpenRouter model {model!r}")
    content = choices[0].message.content
    if not content:
        raise ValueError(f"Empty response from OpenRouter model {model!r}")
    return content


def call_nvidia_nim(
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    timeout: float,
    temperature: float = 1.0,
    reasoning: str | None = None,
) -> str:
    """
    Make one NVIDIA NIM request. Returns non-empty response string or raises.

    Note: NIM does not reliably support response_format=json_object.
    If JSON output is required, the caller must append explicit JSON
    instructions to user_message before calling this function.
    """
    client = _get_nim_client()
    messages = _build_messages(system_prompt, user_message)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
    }
    if reasoning == "low":
        # NIM API reference (nemotron-3-super): thinking is on by default; low_effort shortens the trace
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True, "low_effort": True}}
    response = client.chat.completions.create(**kwargs)
    usage_meter.record_openai_usage("nvidia_nim", model, response)
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise ValueError(f"Empty response from NVIDIA NIM model {model!r}")
    return content


def call_groq(
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    json_mode: bool,
    timeout: float,
    temperature: float = 1.0,
    reasoning: str | None = None,
) -> str:
    """
    Make one Groq request (OpenAI-compatible endpoint). Returns non-empty response string or raises.

    json_mode: if True, sets response_format={"type": "json_object"}; the prompt must still ask for JSON.
    """
    client = _get_groq_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(system_prompt, user_message),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning is not None:
        kwargs["reasoning_effort"] = reasoning  # gpt-oss on Groq: low | medium | high (Groq reasoning docs)
    response = client.chat.completions.create(**kwargs)
    usage_meter.record_openai_usage("groq", model, response)
    choices = response.choices
    if not choices or not choices[0].message.content:
        raise ValueError(f"Empty response from Groq model {model!r}")
    return choices[0].message.content
