"""Public-slice trim gates (ADR-06): no dead prompt builders and no unreferenced constants ship."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "aer"
PROMPT_BUILDERS = PKG / "utils" / "prompt_builders.py"
CONSTANTS = PKG / "utils" / "constants.py"

REMOVED_BUILDERS = frozenset(
    {
        "build_news_prompt",
        "build_postmortem_prompt",
        "build_ig_insight_prompt",
        "build_layoffs_insight_prompt",
    }
)

# Constants with no reader in the package whose boundary or contract is pinned by a test.
# Each entry is debt: wire the constant into the code that owns the threshold, or delete it with its test.
TEST_PINNED_CONSTANTS = frozenset(
    {
        "ATR_VOLATILITY_HIGH",
        "BBL_BREAKOUT_THRESHOLD",
        "GATE_G5_OLLAMA_UPTIME_FLOOR",
        "GROQ_FREE_REQUESTS_PER_DAY",
        "ICHIMOKU_SIGNAL_THRESHOLD",
        "MACD_SIGNAL_THRESHOLD",
        "MOMENTUM_THRESHOLD_WEAK",
        "PREFETCH_EARNINGS_WINDOW_TRADING_DAYS",
        "PREFETCH_EXDATE_WINDOW_TRADING_DAYS",
        "PREFETCH_STRESS_TTL_DIVISOR",
        "REGIME_TO_TEMPLATE_TAG",
        "REQUIRED_SEVERITY_KEY",
        "RSI_OVERBOUGHT",
        "RSI_OVERSOLD",
        "VOLUME_MA_THRESHOLD",
    }
)


def _package_source(excluding: Path) -> str:
    parts = [p.read_text(encoding="utf-8") for p in sorted(PKG.rglob("*.py")) if p.resolve() != excluding.resolve()]
    return "\n".join(parts)


def _mentions(name: str, text: str) -> int:
    return len(re.findall(rf"\b{re.escape(name)}\b", text))


def _module_constants(source: str) -> list[str]:
    names: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            names.append(node.targets[0].id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return [n for n in names if n.isupper()]


def test_removed_prompt_builders_stay_removed():
    defined = {n.name for n in ast.parse(PROMPT_BUILDERS.read_text(encoding="utf-8")).body if hasattr(n, "name")}
    assert not (defined & REMOVED_BUILDERS)


def test_every_prompt_builder_has_a_caller_in_the_package():
    source = PROMPT_BUILDERS.read_text(encoding="utf-8")
    rest = _package_source(excluding=PROMPT_BUILDERS)
    builders = [
        n.name
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("build_") and n.name.endswith("_prompt")
    ]
    assert builders, "no prompt builders found; the gate is reading the wrong file"
    dead = [name for name in builders if _mentions(name, rest) == 0]
    assert dead == [], f"prompt builders with no caller in src/aer: {dead}"


def test_every_constant_has_a_reader():
    source = CONSTANTS.read_text(encoding="utf-8")
    rest = _package_source(excluding=CONSTANTS)
    unread = [
        name
        for name in _module_constants(source)
        if _mentions(name, rest) == 0 and _mentions(name, source) == 1 and name not in TEST_PINNED_CONSTANTS
    ]
    assert unread == [], f"constants no module reads: {unread}"


def test_test_pinned_list_has_no_stale_entry():
    source = CONSTANTS.read_text(encoding="utf-8")
    rest = _package_source(excluding=CONSTANTS)
    defined = set(_module_constants(source))
    missing = sorted(TEST_PINNED_CONSTANTS - defined)
    now_read = sorted(n for n in TEST_PINNED_CONSTANTS & defined if _mentions(n, rest) > 0)
    assert missing == [], f"listed but no longer defined: {missing}"
    assert now_read == [], f"listed but now read by the package, remove from the list: {now_read}"
