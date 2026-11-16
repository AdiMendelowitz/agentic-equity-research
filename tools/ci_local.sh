#!/usr/bin/env bash
# The tests job of .github/workflows/ci.yml on one Python version, plus the redaction scan and licence audit.
# Not run here: the secrets scan, the tracked-files and author checks, and the other Python versions.
# Usage: bash tools/ci_local.sh [python-version]
set -u
PY="${1:-3.14}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}" UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/tmp/uvpy}" UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/aer-venv-$PY}"
export UV_PYTHON="$PY"
cd "$(dirname "$0")/.."
status=0
step() { echo "=== $1"; shift; "$@"; rc=$?; echo "--- exit $rc"; [ $rc -ne 0 ] && status=1; return 0; }
step "python $PY" uv python install "$PY"
step "sync" uv sync --extra dev
step "redaction scan" python3 tools/redact_check.py
step "ruff" uv run ruff check src tests tools docs/report
step "black" uv run black --check --line-length 120 src tests tools docs/report
MYPY_CACHE_DIR=/tmp/aer-mypy-cache step "mypy ceiling" uv run python tools/mypy_ratchet.py
step "collect" uv run pytest --collect-only -q
step "pytest" uv run pytest -m "not live" -q
step "demo smoke" env -u ANTHROPIC_API_KEY -u OPENROUTER_API_KEY -u GROQ_API_KEY -u NVIDIA_NIM_API_KEY uv run aer-demo --tickers 2 --days 300
step "licence audit" uv run pip-licenses --fail-on="GNU General Public License v3 (GPLv3);GNU General Public License v2 (GPLv2);GNU Affero General Public License v3 (AGPLv3);GPLv3;GPLv2;AGPLv3" --summary
echo "=== ci_local overall exit $status"; exit $status
