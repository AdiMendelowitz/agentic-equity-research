#!/usr/bin/env bash
# Mirror of .github/workflows/ci.yml for the bridge shell (no git step). Usage: bash tools/ci_local.sh [python-version]
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
step "ruff" uv run ruff check src tests tools
step "black" uv run black --check --line-length 120 src tests tools
echo "=== mypy (non-blocking)"; MYPY_CACHE_DIR=/tmp/aer-mypy-cache uv run mypy 2>&1 | tail -3
step "collect" uv run pytest --collect-only -q
step "pytest" uv run pytest -m "not live" -q
step "licence audit" uv run pip-licenses --fail-on="GNU General Public License v3 (GPLv3);GNU General Public License v2 (GPLv2);GNU Affero General Public License v3 (AGPLv3);GPLv3;GPLv2;AGPLv3" --summary
echo "=== ci_local overall exit $status"; exit $status
