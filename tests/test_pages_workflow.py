from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
PUBLIC_REPOSITORY = "AdiMendelowitz/agentic-equity-research"


def _pages() -> str:
    return (WORKFLOWS / "pages.yml").read_text(encoding="utf-8")


def _jobs(text: str) -> dict[str, str]:
    body = text.split("\njobs:\n", 1)[1]
    names = re.findall(r"^  (\w+):\n", body, flags=re.M)
    parts = re.split(r"^  \w+:\n", body, flags=re.M)[1:]
    return dict(zip(names, parts, strict=True))


def test_workflow_default_token_is_read_only() -> None:
    head = _pages().split("\njobs:\n", 1)[0]
    assert "permissions:\n  contents: read\n" in head
    assert "write" not in head


def test_write_permissions_belong_to_the_deploy_job_only() -> None:
    jobs = _jobs(_pages())
    assert set(jobs) == {"build", "deploy"}
    assert "write" not in jobs["build"]
    assert re.findall(r"^\s+([\w-]+): write$", jobs["deploy"], flags=re.M) == ["pages", "id-token"]


def test_deploy_runs_in_the_public_repository_only() -> None:
    deploy = _jobs(_pages())["deploy"]
    assert f"if: github.repository == '{PUBLIC_REPOSITORY}'" in deploy
    assert "needs: build" in deploy
    assert "name: github-pages" in deploy


def test_scan_of_the_built_site_precedes_the_upload() -> None:
    build = _jobs(_pages())["build"]
    order = [
        "docs/report/build_figures.py",
        "docs/report/build_report.py --out site",
        "tools/redact_check.py site",
        "actions/upload-pages-artifact@",
    ]
    positions = [build.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "path: site" in build


def test_workflow_uses_no_secret() -> None:
    assert "secrets." not in _pages()


def test_every_workflow_pins_its_runner_image() -> None:
    files = sorted(WORKFLOWS.glob("*.yml"))
    assert [p.name for p in files] == ["ci.yml", "pages.yml"]
    for path in files:
        runners = re.findall(r"runs-on: (\S+)", path.read_text(encoding="utf-8"))
        assert runners and set(runners) == {"ubuntu-24.04"}, path.name
