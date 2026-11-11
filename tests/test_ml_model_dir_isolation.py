"""(2026-09-28): the test suite wrote AAPL_<date>.joblib and its lock into the production ml_models/ directory
on every gate day, and _cleanup_old_models pruned real files there. conftest now points MODEL_DIR at tmp_path for
every test."""

from __future__ import annotations

from pathlib import Path

import aer.ml.constants as mlc
import aer.ml.models as mlm

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_model_dir_is_not_the_production_directory():
    assert mlc.MODEL_DIR != _REPO_ROOT / "ml_models"
    assert mlm.MODEL_DIR != _REPO_ROOT / "ml_models"


def test_both_modules_share_one_redirected_directory(tmp_path):
    assert mlm.MODEL_DIR == mlc.MODEL_DIR
    assert tmp_path in mlc.MODEL_DIR.parents


def test_the_pruner_only_touches_the_redirected_directory(tmp_path):
    """_cleanup_old_models globs MODEL_DIR for <ticker>_*.joblib; a stale file there is the only one it may see."""
    mlm.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stale = mlm.MODEL_DIR / "ZZISO_2000-01-01.joblib"
    stale.write_bytes(b"x")
    import os
    import time

    old = time.time() - 30 * 86400
    os.utime(stale, (old, old))
    mlm._cleanup_old_models("ZZISO")
    assert not stale.exists()
    assert not (_REPO_ROOT / "ml_models" / "ZZISO_2000-01-01.joblib").exists()
