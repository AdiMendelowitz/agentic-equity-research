<#
.SYNOPSIS
  One-command local check for agentic-equity-research, run by Adi. Mirrors the CI steps and saves the full output.
.DESCRIPTION
  Usage: powershell -File tools\check.ps1            (add -SkipSync to skip the dependency sync)
  Runs, in the project environment (.venv, Python from .python-version): dependency sync, redaction scan, ruff, black,
  mypy (non-blocking), figure build, test suite, and the demo smoke test once the demo package exists.
  Every line of output is shown and also written, UTF-8, to notes\runs\check-<timestamp>.log together with the commit id,
  so the log is the record of the run. Exit code 0 means every blocking step passed.
#>
param([switch]$SkipSync)
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -Path $repo
$runs = Join-Path $repo "notes\runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $runs "check-$stamp.log"
$utf8 = New-Object System.Text.UTF8Encoding $false
# native output is decoded with the console code page unless both sides agree on UTF-8
try { [Console]::OutputEncoding = $utf8 } catch { Write-Host "console encoding unchanged: $_" }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$failed = New-Object System.Collections.Generic.List[string]
$summary = New-Object System.Collections.Generic.List[string]

function Write-Log([string[]]$lines) {
  foreach ($l in $lines) { Write-Host $l }
  [System.IO.File]::AppendAllLines($log, [string[]]$lines, $utf8)
}

function Invoke-Step([string]$name, [string]$commandLine, [bool]$blocking) {
  Write-Log @("", "=== $name", "> $commandLine")
  $started = Get-Date
  # cmd merges stderr into stdout, so PowerShell 5.1 never turns a progress line into an error record
  cmd /c "$commandLine 2>&1" | ForEach-Object { Write-Log @([string]$_) }
  $code = $LASTEXITCODE
  $secs = [int]((Get-Date) - $started).TotalSeconds
  $kind = "blocking"
  if (-not $blocking) { $kind = "non-blocking" }
  Write-Log @("--- exit $code after $secs s ($kind)")
  $script:summary.Add(("{0,-18} exit {1,-3} {2,5} s  {3}" -f $name, $code, $secs, $kind))
  if ($blocking -and $code -ne 0) { $script:failed.Add($name) }
}

$head = (git --no-pager rev-parse --short HEAD)
$dirty = @(git --no-pager status --short).Count
Write-Log @("check started $stamp", "commit $head, $dirty uncommitted path(s)", "folder $repo", "host $env:COMPUTERNAME, PowerShell $($PSVersionTable.PSVersion)")

if (-not $SkipSync) { Invoke-Step "sync" "uv sync --extra dev" $true }
Invoke-Step "python" "uv run python -V" $true
Invoke-Step "redaction scan" "uv run python tools/redact_check.py" $true
Invoke-Step "ruff" "uv run ruff check src tests tools docs/report" $true
Invoke-Step "black" "uv run black --check --line-length 120 src tests tools docs/report" $true
Invoke-Step "mypy" "uv run mypy" $false
if (Test-Path -Path "docs\report\build_figures.py") { Invoke-Step "figures" "uv run python docs/report/build_figures.py" $true }
Invoke-Step "collect" "uv run pytest --collect-only -q" $true
Invoke-Step "pytest" "uv run pytest" $true
if (Test-Path -Path "src\aer\demo\cli.py") {
  Invoke-Step "demo smoke" "uv run aer-demo --tickers 2 --days 300" $true
}
if (Test-Path -Path "tools\quickstart.ps1") { Invoke-Step "quick start" "powershell -NoProfile -File tools\quickstart.ps1" $true }

Write-Log (@("", "=== summary") + $summary.ToArray())
if ($failed.Count -eq 0) {
  Write-Log @("RESULT: PASS (every blocking step exited 0)", "log: $log")
  exit 0
} else {
  Write-Log @("RESULT: FAIL in: $($failed -join ', ')", "log: $log")
  exit 1
}
