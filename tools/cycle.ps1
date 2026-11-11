<#
.SYNOPSIS
  One cycle of local verification for agentic-equity-research: optional lock refresh, the full check, and on PASS the named landing.
.DESCRIPTION
  Usage (by hand):   powershell -File tools\cycle.ps1 -Action land -Landing 2026-11-11-demo.txt [-Lock]
  Actions: check        run tools\check.ps1
           land         run tools\check.ps1 and, only if it passes, tools\land.ps1 for the named file under notes\landing
           control-one  time one cell of the control experiment (docs/report/run_control.py --one-cell)
           control      run the full control experiment (docs/report/run_control.py)
  -Lock runs `uv lock` first (needed after a dependency change).
  The formatter and the linter's safe fixes (black, ruff --fix) are applied to src, tests, tools and docs/report before the
  check; -NoFix skips that.
  Everything is logged under notes\runs\ and a short result is written to -ResultFile when given.
  This script runs only the fixed commands above. tools\watch.ps1 calls it for requests Adi has authorised by starting the watcher.
#>
param(
  [ValidateSet("check", "land", "control-one", "control")][string]$Action = "check",
  [string]$Landing = "",
  [switch]$Lock,
  [switch]$NoFix,
  [string]$ResultFile = ""
)
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -Path $repo
$runs = Join-Path $repo "notes\runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$utf8 = New-Object System.Text.UTF8Encoding $false
$result = New-Object System.Collections.Generic.List[string]

function Add-Result([string]$line) {
  Write-Host $line
  $script:result.Add($line)
}

function Invoke-Logged([string]$commandLine, [string]$logPath) {
  [System.IO.File]::AppendAllLines($logPath, [string[]]@("> $commandLine"), $utf8)
  cmd /c "$commandLine 2>&1" | ForEach-Object {
    $line = [string]$_
    Write-Host $line
    [System.IO.File]::AppendAllLines($logPath, [string[]]@($line), $utf8)
  }
  return $LASTEXITCODE
}

function Complete-Cycle([bool]$passed) {
  if ($passed) { Add-Result "RESULT: PASS" } else { Add-Result "RESULT: FAIL" }
  Add-Result ("cycle finished " + (Get-Date -Format "yyyyMMdd-HHmmss"))
  if ($ResultFile -ne "") { [System.IO.File]::WriteAllLines($ResultFile, $script:result.ToArray(), $utf8) }
  if ($passed) { exit 0 } else { exit 1 }
}

$head = (git --no-pager rev-parse --short HEAD)
Add-Result "cycle started $stamp action=$Action landing=$Landing lock=$($Lock.IsPresent) fix=$(-not $NoFix.IsPresent) commit=$head"

# Claude lists the sha256 of every file it wrote for this request; a mismatch means a write did not reach this disk.
if ($ResultFile -ne "") {
  $requestPath = Join-Path $runs ("requests\" + [System.IO.Path]::GetFileNameWithoutExtension($ResultFile) + ".request")
  if (Test-Path -Path $requestPath) {
    $checked = 0
    $stale = 0
    foreach ($line in (Get-Content -Path $requestPath)) {
      if ($line -match '^sha256:([A-Za-z0-9_./-]+)=([0-9a-fA-F]{64})$') {
        $rel = $Matches[1]
        $want = $Matches[2].ToLower()
        $full = Join-Path $repo ($rel -replace '/', '\')
        if ($rel.Contains("..") -or -not (Test-Path -Path $full)) {
          Add-Result "WRITE CHECK: missing or rejected path $rel"
          $stale++
        } else {
          $have = (Get-FileHash -Algorithm SHA256 -Path $full).Hash.ToLower()
          $checked++
          if ($have -ne $want) {
            Add-Result "WRITE CHECK: $rel is not the content that was written (expected $want, found $have)"
            $stale++
          }
        }
      }
    }
    Add-Result "write verification: $checked file(s) hashed, $stale problem(s)"
    if ($stale -gt 0) { Complete-Cycle $false }
  }
}

if ($Action -eq "land") {
  if ($Landing -notmatch '^[A-Za-z0-9._-]+\.txt$') { Add-Result "landing name rejected: '$Landing'"; Complete-Cycle $false }
  if (-not (Test-Path -Path (Join-Path $repo "notes\landing\$Landing"))) { Add-Result "landing file not found: notes\landing\$Landing"; Complete-Cycle $false }
}

if ($Lock) {
  $log = Join-Path $runs "lock-$stamp.log"
  $code = Invoke-Logged "uv lock" $log
  Add-Result "lock exit $code log $log"
  if ($code -ne 0) { Complete-Cycle $false }
}

if (-not $NoFix) {
  $log = Join-Path $runs "fix-$stamp.log"
  $code = Invoke-Logged "uv sync --extra dev" $log
  if ($code -ne 0) { Add-Result "sync before fix exit $code log $log"; Complete-Cycle $false }
  $code = Invoke-Logged "uv run ruff check --fix src tests tools docs/report" $log
  Add-Result "ruff fix exit $code (non-zero means findings remain; the check decides) log $log"
  $code = Invoke-Logged "uv run black --line-length 120 src tests tools docs/report" $log
  Add-Result "black format exit $code log $log"
}

if ($Action -eq "check" -or $Action -eq "land") {
  & powershell -NoProfile -File (Join-Path $repo "tools\check.ps1")
  $code = $LASTEXITCODE
  $latest = Get-ChildItem -Path $runs -Filter "check-*.log" | Sort-Object LastWriteTime | Select-Object -Last 1
  Add-Result "check exit $code log $($latest.FullName)"
  if ($code -ne 0) { Complete-Cycle $false }
}

if ($Action -eq "land") {
  $log = Join-Path $runs "landfull-$stamp.log"
  $code = Invoke-Logged "powershell -NoProfile -File tools\land.ps1 -Landing notes\landing\$Landing -NonInteractive" $log
  Add-Result "land exit $code (0 means committed, pushed and CI green) log $log"
  if ($code -ne 0) { Complete-Cycle $false }
}

if ($Action -eq "control-one" -or $Action -eq "control") {
  $log = Join-Path $runs "control-$stamp.log"
  $commandLine = "uv run python docs/report/run_control.py"
  if ($Action -eq "control-one") { $commandLine = "$commandLine --one-cell" }
  $code = Invoke-Logged $commandLine $log
  Add-Result "$Action exit $code log $log"
  if ($code -ne 0) { Complete-Cycle $false }
}

Complete-Cycle $true
