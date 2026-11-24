"""Every file under docs/data matches its pinned sha256."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs" / "data"

# Hashed with CRLF replaced by LF: .gitattributes sets eol=lf, so a Windows working tree can hold CRLF where a CI
# checkout holds LF. A change to a pinned file is a data correction; its new value lands with the dated correction note.
PINS = {
    "detection_curve.csv": "9c613251bd1efc3a85d7c15bfd583848aae63685466ae21d9107ab196fdd9a83",
    "detection_curve_run1.csv": "202713abf67d50cc8b18c677ff62552118959b29324ee67b4f9aa55eb6868ce3",
    "detection_curve_v3.csv": "0a50a1d5272254ca935deb3e41ddadd87e82e5724a0a5c70ac1ded32def3b9f2",
    "eval_backtest_sharpe.csv": "2d141a345586dce7f13fbafbc678e1bd1504c7c24311400849a42c8645d29eda",
    "eval_calibration.csv": "bc7b05b5e6e2c5221ee19c05bc1bb14ebf5d801a61b6b57f0b21e6cc6dc05d6b",
    "eval_ml_gate.csv": "9ef46be9d28817b1bcc24859db35f7788d58047d743f9936d32532a334f34ff4",
    "eval_recommendation.csv": "65de7817226e5bd084b39baf19644b2dc946f7c33c46b9324d909d991b748afd",
    "eval_sample.csv": "c75b3c054ddac84b96626b3dd8c7aee2722e9cd2564f58fac0e37dcfe53ef7c7",
    "exploratory_zero_signal_v3.csv": "d9fc262968f8ddd860ef6118c56c41a8b1f5541b21f9e0f57887003d593a8e17",
    "llm_usage.jsonl": "d5c7eddc6c30a4075f8d4123ea1e40d46dd2ccd1d22e105857c0883b2249aff3",
    "run_history.csv": "7ff05e608df31d025160e0ab2caa3d05a22e2fd9022f428ff6836b83b8270c1c",
}


def _normalised_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_pins_cover_exactly_the_files_on_disk() -> None:
    assert sorted(p.relative_to(DATA_DIR).as_posix() for p in DATA_DIR.rglob("*") if p.is_file()) == sorted(PINS)


def test_normalisation_ignores_line_endings_and_catches_content(tmp_path: Path) -> None:
    lf, crlf, edited = tmp_path / "lf.csv", tmp_path / "crlf.csv", tmp_path / "edited.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    crlf.write_bytes(b"a,b\r\n1,2\r\n")
    edited.write_bytes(b"a,b\n1,3\n")
    assert _normalised_sha256(lf) == _normalised_sha256(crlf)
    assert _normalised_sha256(lf) != _normalised_sha256(edited)


@pytest.mark.parametrize("name", sorted(PINS))
def test_data_file_matches_its_pin(name: str) -> None:
    assert _normalised_sha256(DATA_DIR / name) == PINS[name]
