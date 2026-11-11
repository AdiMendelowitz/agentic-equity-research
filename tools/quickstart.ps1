<#
.SYNOPSIS
  Fresh-environment quick start check, run by Adi: proves the built wheel works outside the repository.
.DESCRIPTION
  Usage: powershell -File tools\quickstart.ps1                 (checks the wheel built from this folder)
         powershell -File tools\quickstart.ps1 -Source <dir>   (checks another checkout, for example a fresh clone)
  Builds the wheel, installs it into a new virtual environment under %TEMP% (Python 3.14 by default), changes to an empty
  folder outside the repository so nothing is imported from the source tree, and runs the README command:
  aer-demo. Every line is written, UTF-8, to notes\runs\quickstart-<timestamp>.log.
  Exit code 0 means the demo command exited 0 from the installed wheel.
#>
param([string]$Source = "", [string]$Python = "3.14")
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ($Source -eq "") { $Source = $repo }
$Source = (Resolve-Path -Path $Source).Path
$runs = Join-Path $repo "notes\runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $runs "quickstart-$stamp.log"
$utf8 = New-Object System.Text.UTF8Encoding $false
# native output is decoded with the console code page unless both sides agree on UTF-8
try { [Console]::OutputEncoding = $utf8 } catch { Write-Host "console encoding unchanged: $_" }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$work = Join-Path $env:TEMP "aer-quickstart-$stamp"
$dist = Join-Path $work "dist"
$venv = Join-Path $work "venv"
$empty = Join-Path $work "empty"
if ($work -match "\s") { throw "the temp folder path contains a space, which this script does not quote: $work" }
New-Item -ItemType Directory -Force -Path $dist, $empty | Out-Null
$failed = New-Object System.Collections.Generic.List[string]

function Write-Log([string[]]$lines) {
  foreach ($l in $lines) { Write-Host $l }
  [System.IO.File]::AppendAllLines($log, [string[]]$lines, $utf8)
}

function Invoke-Step([string]$name, [string]$commandLine) {
  Write-Log @("", "=== $name", "> $commandLine")
  cmd /c "$commandLine 2>&1" | ForEach-Object { Write-Log @([string]$_) }
  $code = $LASTEXITCODE
  Write-Log @("--- exit $code")
  if ($code -ne 0) { $script:failed.Add($name) }
}

Write-Log @("quickstart started $stamp", "source $Source", "work folder $work", "python $Python")
Set-Location -Path $Source
Invoke-Step "build wheel" "uv build --wheel --out-dir $dist"
$wheel = Get-ChildItem -Path $dist -Filter "*.whl" | Select-Object -First 1
if (-not $wheel) {
  Write-Log @("RESULT: FAIL (no wheel was built)", "log: $log")
  exit 1
}
Write-Log @("wheel $($wheel.Name), $($wheel.Length) bytes")
Invoke-Step "create environment" "uv venv --python $Python $venv"
$py = Join-Path $venv "Scripts\python.exe"
$demo = Join-Path $venv "Scripts\aer-demo.exe"
Invoke-Step "install wheel" "uv pip install --python $py $($wheel.FullName)"
Set-Location -Path $empty
Invoke-Step "aer-demo" "$demo"
Set-Location -Path $repo

if ($failed.Count -eq 0) {
  Write-Log @("", "RESULT: PASS (wheel installed and the demo command exited 0 outside the repository)", "log: $log")
  exit 0
} else {
  Write-Log @("", "RESULT: FAIL in: $($failed -join ', ')", "log: $log")
  exit 1
}
