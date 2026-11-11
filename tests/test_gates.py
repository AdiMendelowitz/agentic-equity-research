"""Architecture gate tests for the aer package layout (src/aer, tests under tests/)."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "aer"


def test_graph_invocation_allowlist():
    """Only aer/graph.py and aer/run_pipeline.py invoke the graph or build_brief."""
    allowlist = {PKG / "graph.py", PKG / "run_pipeline.py"}
    offenders = []
    for py_file in PKG.rglob("*.py"):
        if py_file in allowlist:
            continue
        if re.search(r"graph\.invoke\(|build_brief\(", py_file.read_text(encoding="utf-8")):
            offenders.append(str(py_file.relative_to(PKG)))
    assert offenders == [], f"Unauthorised graph invocation in {offenders}"


def test_pages_import_no_llm_client():
    """aer/pages (when present) imports no LLM client."""
    pages_dir = PKG / "pages"
    if not pages_dir.is_dir():
        return
    patterns = [
        r"from.*llm_client",
        r"import.*llm_client",
        r"from.*anthropic",
        r"import.*anthropic",
        r"from.*ollama",
        r"import.*ollama",
    ]
    for py_file in pages_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for pattern in patterns:
            assert not re.search(pattern, content), f"{pattern} found in {py_file}"


def test_sqlite_connect_centralised_in_db_module():
    """Raw sqlite3.connect appears in no aer module except aer/utils/db.py."""
    db_module = PKG / "utils" / "db.py"
    offenders = [
        str(py_file.relative_to(PKG))
        for py_file in PKG.rglob("*.py")
        if py_file != db_module and re.search(r"sqlite3\.connect\(", py_file.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"raw sqlite3.connect outside db.py: {offenders}"


def test_gitignore_excludes_internal_files():
    """.gitignore lists notes/ and CLAUDE.md so internal controls are never published."""
    lines = {ln.strip() for ln in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert "notes/" in lines
    assert "CLAUDE.md" in lines


EVAL_HEADERS = {
    "eval_backtest_sharpe.csv": ["sharpe_bin", "n_rows"],
    "eval_calibration.csv": ["conviction_bin", "n_resolved", "n_correct"],
    "eval_ml_gate.csv": ["cv_f1_bin", "n_rows", "n_skipped", "n_ml_resolved", "n_ml_correct"],
    "eval_recommendation.csv": ["recommendation", "n_rows", "n_resolved", "n_synth_correct"],
    "eval_sample.csv": ["metric", "value"],
}


def _header(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()[0].split(",")


def test_aggregate_schemas():
    """Published aggregates keep their exact headers and expose no per-ticker or per-date column."""
    data_dir = ROOT / "docs" / "data"
    assert {p.name for p in data_dir.glob("eval_*.csv")} == set(EVAL_HEADERS)
    for name, expected in EVAL_HEADERS.items():
        assert _header(data_dir / name) == expected, name
        for column in expected:
            assert not any(word in column.lower() for word in ("ticker", "symbol", "week", "date")), (name, column)
    assert _header(data_dir / "run_history.csv") == ["run_date", "outcome", "exit_code"]
    assert _header(data_dir / "detection_curve.csv") == [
        "signal",
        "seed",
        "gate_predicted",
        "gate_reason",
        "cv_f1_mean",
        "cv_f1_std",
        "model_balanced_accuracy",
        "oracle_balanced_accuracy",
        "majority_accuracy",
        "label_base_rate",
        "n_oos",
    ]
    assert _header(data_dir / "detection_curve_run1.csv") == [
        "signal",
        "seed",
        "gate_predicted",
        "cv_f1_mean",
        "oos_accuracy",
        "oracle_accuracy",
        "n_oos",
    ]
    lines = (data_dir / "llm_usage.jsonl").read_text(encoding="utf-8").splitlines()
    for line in filter(str.strip, lines):
        record = json.loads(line)
        assert not any(("ticker" in key.lower() or "symbol" in key.lower()) for key in record), line[:80]


def test_redaction_scan_clean(monkeypatch, capsys):
    """tools/redact_check.py reports no hit over the whole tree."""
    spec = importlib.util.spec_from_file_location("redact_check", ROOT / "tools" / "redact_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(ROOT)
    code = module.main(["redact_check.py"])
    assert code == 0, capsys.readouterr().out[-2000:]


def test_demo_never_reaches_ml_or_evaluation():
    """The demo package is unreachable from ML, evaluation and outcome labelling."""
    files = [*(PKG / "ml").rglob("*.py"), *(PKG / "evaluation").rglob("*.py"), PKG / "utils" / "outcome_labeler.py"]
    assert files
    offenders = [
        (str(f.relative_to(PKG)), token)
        for f in files
        for token in ("aer.demo", "demo_only")
        if token in f.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_renderers_use_shared_banner():
    """Every demo or page renderer goes through the shared banner."""
    demo = PKG / "demo"
    if not demo.is_dir():
        pytest.skip("aer.demo lands at M2; the banner gate arms itself when the package exists")
    banner = demo / "banner.py"
    assert banner.is_file()
    assert "def render_banner(" in banner.read_text(encoding="utf-8")
    offenders = []
    for folder in (demo, PKG / "pages"):
        if not folder.is_dir():
            continue
        for py_file in folder.rglob("*.py"):
            if py_file == banner:
                continue
            source = py_file.read_text(encoding="utf-8")
            if ("def render_" in source or "def main(" in source) and "render_banner" not in source:
                offenders.append(str(py_file.relative_to(PKG)))
    assert offenders == []


EXCLUDED_MODULES: tuple[str, ...] = (
    "agents/insider_buying.py",
    "agents/instagram_agent.py",
    "agents/layoffs_agent.py",
    "agents/news.py",
    "agents/options_strategy.py",
    "agents/options_utils.py",
    "agents/pelosi_agent.py",
    "agents/professor/__init__.py",
    "agents/professor/professor.py",
    "agents/social_synthesis.py",
    "alerts/__init__.py",
    "alerts/engine.py",
    "alerts/models.py",
    "alerts/pm_divergence.py",
    "alerts/social_consensus.py",
    "backtesting/__init__.py",
    "backtesting/harness.py",
    "backtesting/metrics.py",
    "backtesting/strategy.py",
    "catalyst/__init__.py",
    "catalyst/models.py",
    "catalyst/parser.py",
    "catalyst/writer.py",
    "data/cot.py",
    "data/finviz.py",
    "data/lseg.py",
    "data/pelosi_source.py",
    "data/prediction_markets.py",
    "data/pull_audit.py",
    "data/social_intelligence.py",
    "evaluation/alt_data_eval.py",
    "export.py",
    "pages/alerts.py",
    "pages/discovery.py",
    "pages/market_overview.py",
    "pages/pelosi_tracker.py",
    "pages/portfolio.py",
    "pages/wiki.py",
    "portfolio/__init__.py",
    "portfolio/analytics.py",
    "portfolio/models.py",
    "prefetch/corp_action.py",
    "prefetch/earnings_window.py",
    "prefetch/enforcement.py",
    "prefetch/stress.py",
    "prefetch/vintage.py",
    "simulation/backtest_simulation.py",
    "simulation/blind_investor.py",
    "simulation/charting.py",
    "simulation/csv_writer.py",
    "simulation/postmortem.py",
    "simulation/true_investor.py",
    "thesis/__init__.py",
    "thesis/models.py",
    "thesis/parser.py",
    "thesis/writer.py",
    "utils/amount_range.py",
    "utils/autonomous_debug.py",
    "utils/calibration_ids.py",
    "utils/calibration_orchestrator.py",
    "utils/extraction_prompt_builders.py",
    "utils/final_gate.py",
    "utils/lseg_prefetch.py",
    "utils/lseg_signals.py",
    "utils/lseg_status.py",
    "utils/morning_note.py",
    "utils/portfolio_monitor.py",
    "utils/professor_prompt_builders.py",
    "utils/professor_utils.py",
    "utils/promotion.py",
    "utils/protected_files.py",
    "utils/sector_scan.py",
    "utils/social_dedup.py",
    "utils/sqlite_store.py",
)


def test_excluded_modules_absent():
    """Modules excluded from the port do not exist, and no module imports an excluded third-party package."""
    present = [rel for rel in EXCLUDED_MODULES if (PKG / rel).exists()]
    assert present == []
    pattern = re.compile(r"^\s*(from|import)\s+(instaloader|playwright|predictit|discord|backtrader)\b", re.M)
    offenders = [str(p.relative_to(PKG)) for p in PKG.rglob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
