<#
.SYNOPSIS
  One-command landing for agentic-equity-research: check, stage the paths listed in a landing file, commit with its message, push, wait for CI.
.DESCRIPTION
  Usage: powershell -File tools\land.ps1 -Landing notes\landing\<id>.txt
  Landing file format: first line = commit message; remaining lines = pathspecs to stage (one per line, # comments allowed).
  Refuses to run if notes/ or CLAUDE.md would be staged, if the author email is not a GitHub noreply address,
  if gitleaks (when installed) reports a finding, or if a local pre-check fails (package metadata build and test collection,
  run with uv in an environment outside the repository). Asks for a yes when more than 20 files are staged.
  After the push it waits for the CI run of the new commit and prints its result (when gh is installed).
  -SkipPreChecks skips the uv build and collection checks only. -NonInteractive refuses a landing of more than 20 files
  instead of asking. The exit code is the CI result (0 when CI is green or when no CI run was waited for).
#>
param(
  [Parameter(Mandatory = $true)][string]$Landing,
  [switch]$Push = $true,
  [switch]$SkipPreChecks,
  [switch]$NonInteractive
)
$ErrorActionPreference = "Stop"
$ciCode = 0
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -Path $repo
$runs = Join-Path $repo "notes\runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
try { Start-Transcript -Path (Join-Path $runs ("land-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".log")) | Out-Null } catch { Write-Output "transcript not started: $_" }
if (-not (Test-Path -Path $Landing)) { throw "landing file not found: $Landing" }
$lines = Get-Content -Path $Landing | Where-Object { $_.Trim() -ne "" -and -not $_.Trim().StartsWith("#") }
$message = $lines[0]
$paths = $lines | Select-Object -Skip 1
if ($paths.Count -eq 0) { throw "landing file lists no paths" }
foreach ($p in $paths) { if ($p -match '^(notes([\\/]|$)|CLAUDE\.md$)') { throw "refusing to stage internal path: $p" } }
$email = git --no-pager config --get user.email
if ($email -notmatch '^\d+\+[A-Za-z0-9-]+@users\.noreply\.github\.com$') { throw "author email is not a GitHub noreply address: $email" }
# the private history must never reach the public repository, which now carries the old name
$origin = git --no-pager remote get-url origin
if ($origin -notmatch 'agentic-equity-research-dev(\.git)?$') { throw "origin is not the private -dev repository: $origin; landing refused" }
if (Get-Command gitleaks -ErrorAction SilentlyContinue) {
  # scan what this landing publishes; ignored caches in the folder never reach a commit and CI scans every pushed commit
  foreach ($p in $paths) {
    # a deleted path (the old name of a renamed file) has nothing to scan
    if (-not (Test-Path -Path $p)) { Write-Output "gitleaks: skipped deleted path $p"; continue }
    gitleaks dir $p --no-banner --redact --verbose
    if ($LASTEXITCODE -ne 0) { throw "gitleaks reported a finding under $p; landing refused" }
  }
} else { Write-Output "gitleaks not installed locally; CI runs it on push" }
python tools/redact_check.py
if ($LASTEXITCODE -ne 0) { throw "redaction scan reported a hit; landing refused" }
if ($SkipPreChecks) {
  Write-Output "pre-checks skipped by switch"
} elseif (Get-Command uv -ErrorAction SilentlyContinue) {
  $env:UV_PROJECT_ENVIRONMENT = Join-Path $env:LOCALAPPDATA "aer-landing-venv"
  $buildDir = Join-Path $env:TEMP "aer-landing-build"
  Write-Output "pre-check: package metadata build (first run downloads dependencies and can take a few minutes)"
  uv build --out-dir $buildDir
  if ($LASTEXITCODE -ne 0) { throw "uv build failed; landing refused (rerun with -SkipPreChecks only if the failure is in the check itself)" }
  Write-Output "pre-check: test collection"
  # same interpreter as the first CI job; an unpinned run takes the newest Python on the machine
  uv run --python 3.14 --extra dev pytest --collect-only -q
  if ($LASTEXITCODE -ne 0) { throw "test collection failed; landing refused (rerun with -SkipPreChecks only if the failure is in the check itself)" }
  Remove-Item Env:\UV_PROJECT_ENVIRONMENT
} else { Write-Output "uv not installed locally; pre-checks skipped, CI runs them on push" }
foreach ($p in $paths) { git --no-pager add -- $p }
$staged = @(git --no-pager diff --cached --name-only)
if ($staged.Count -eq 0) { throw "nothing staged" }
foreach ($s in $staged) { if ($s -match '^(notes/|CLAUDE\.md$)') { git --no-pager reset --quiet; throw "internal file staged; landing refused: $s" } }
Write-Output "staged:"; Write-Output $staged
if ($staged.Count -gt 20 -and $NonInteractive) {
  git --no-pager reset --quiet
  throw "$($staged.Count) files staged, above the 20-file limit for an unattended landing; run tools\land.ps1 yourself to confirm"
}
if ($staged.Count -gt 20) {
  $answer = Read-Host "$($staged.Count) files staged; type yes to commit"
  if ($answer -ne "yes") { git --no-pager reset --quiet; throw "landing cancelled; nothing committed" }
}
git --no-pager commit --message $message | ForEach-Object { Write-Output "$_" }
if ($LASTEXITCODE -ne 0) { throw "commit failed" }
Write-Output "commit: $(git --no-pager rev-parse --short HEAD)"
if ($Push) { git --no-pager push; if ($LASTEXITCODE -ne 0) { throw "push failed" } }
Write-Output "authors:"; git --no-pager log --format="%ae" | Sort-Object -Unique | ForEach-Object { Write-Output "$_" }
Write-Output "head:"; git --no-pager log --oneline --max-count 3 | ForEach-Object { Write-Output "$_" }
if ($Push -and (Get-Command gh -ErrorAction SilentlyContinue)) {
  $sha = git --no-pager rev-parse HEAD
  $runId = $null
  foreach ($attempt in 1..8) {
    Start-Sleep -Seconds 5
    $runId = gh run list --commit $sha --limit 1 --json databaseId --jq ".[0].databaseId"
    if ($runId) { break }
  }
  gh run list --limit 4 | ForEach-Object { Write-Output "$_" }
  Write-Output "ci run: $runId"
  if ($runId) {
    Write-Output "waiting for CI run $runId (about two minutes)"
    gh run watch $runId --exit-status
    $ciCode = $LASTEXITCODE
    Write-Output "ci exit code: $ciCode (0 means green)"
  } else { Write-Output "no CI run found yet for $sha; check gh run list in a minute" }
}
try { Stop-Transcript | Out-Null } catch { }
exit $ciCode
