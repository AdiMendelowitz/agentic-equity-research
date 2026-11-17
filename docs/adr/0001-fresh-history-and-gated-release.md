# ADR 0001: Fresh history and gated release

## Status

Accepted

## Context

The code in this repository grew in a private development history that holds working notes and other material meant to stay internal. A reader of a public repository can inspect every commit, tag and workflow log, so anything left in the history is published.

## Decision

This repository has no inherited history. Every change is checked locally first. A release script exports the files tracked at the head of a private repository and refuses to continue if the export contains internal paths, a gitleaks finding, a redaction-scan hit or a failed fresh-environment quick start. The owner then confirms at a prompt, and the export is pushed to this repository as one new commit. Public CI then checks it again on Python 3.12, 3.13 and 3.14. CI also runs gitleaks, the redaction scan and a licence audit, rejects tracked internal files, and rejects any commit author that is not a GitHub noreply address.

## Alternatives considered

- **Rewrite the old history with a filter tool.** Residue in reflogs, tags and large objects is hard to prove absent, so a reader could not be given a firm assurance.
- **Make the private repository public after rotating secrets.** This would publish its full history and workflow logs.
- **Push full history after scanning only the current tree.** Files removed in earlier commits would stay visible in history.

## Consequences

- Positive. Each release is audited as the exact set of files a reader receives, and public CI repeats the scans.
- Negative. This repository shows one commit per release, so the order in which early work was written cannot be checked here. The README states this limit for control versions 1 and 2.

## Enforced by

- `tests/test_gates.py` (`test_redaction_scan_clean`, `test_gitignore_excludes_internal_files`)
- `tests/test_redact_site.py`
- `tests/test_pages_workflow.py`
- `.github/workflows/ci.yml` and `tools/release.ps1` (checks that run outside the test suite)
