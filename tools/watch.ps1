<#
.SYNOPSIS
  Watcher for agentic-equity-research. Adi starts it once; it then runs tools\cycle.ps1 for each request file that appears.
.DESCRIPTION
  Usage: powershell -File tools\watch.ps1        (leave the window open; close it or press Ctrl+C to stop)
  Starting this script is Adi's authorisation for the runs it performs. It does exactly one thing: for each file
  notes\runs\requests\*.request it calls tools\cycle.ps1 with the action named in the file, then moves the request to
  notes\runs\requests\processed\ and leaves the outcome in notes\runs\results\<name>.result.
  A request file holds lines of the form key=value. Keys: action (check, land, control-one, control), landing (a file name
  under notes\landing, required for land), lock (true to run uv lock first), fix (false to skip the black and ruff --fix step that otherwise runs first). Anything else is rejected and nothing runs.
  The watcher never changes repository visibility, never runs a command taken from a request, and never lands more than
  20 files without Adi (land.ps1 refuses in that case and asks for a manual run).
#>
$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -Path $repo
$requests = Join-Path $repo "notes\runs\requests"
$processed = Join-Path $requests "processed"
$results = Join-Path $repo "notes\runs\results"
foreach ($d in @($requests, $processed, $results)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
$alive = Join-Path $repo "notes\runs\watcher.alive"
$cycle = Join-Path $repo "tools\cycle.ps1"
$utf8 = New-Object System.Text.UTF8Encoding $false
$allowed = @("check", "land", "control-one", "control")
Write-Host "watcher started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') in $repo"
Write-Host "watching notes\runs\requests for *.request; close this window or press Ctrl+C to stop"

while ($true) {
  [System.IO.File]::WriteAllText($alive, (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $utf8)
  $next = Get-ChildItem -Path $requests -Filter "*.request" -File | Sort-Object Name | Select-Object -First 1
  if ($next) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($next.Name)
    $resultPath = Join-Path $results "$name.result"
    $fields = @{}
    foreach ($line in (Get-Content -Path $next.FullName)) {
      $parts = $line.Split("=", 2)
      if ($parts.Count -eq 2) { $fields[$parts[0].Trim().ToLower()] = $parts[1].Trim() }
    }
    $action = [string]$fields["action"]
    $landing = [string]$fields["landing"]
    $lock = ([string]$fields["lock"]).ToLower() -eq "true"
    $fix = ([string]$fields["fix"]).ToLower() -ne "false"
    $reason = ""
    if ($allowed -notcontains $action) { $reason = "unknown action '$action'" }
    elseif ($action -eq "land" -and $landing -notmatch '^[A-Za-z0-9._-]+\.txt$') { $reason = "landing name rejected: '$landing'" }
    elseif ($action -ne "land" -and $landing -ne "") { $reason = "landing given for an action that does not land" }
    Write-Host ""
    Write-Host "=== request $($next.Name) at $(Get-Date -Format 'HH:mm:ss'): action=$action landing=$landing lock=$lock fix=$fix"
    if ($reason -ne "") {
      Write-Host "rejected: $reason"
      [System.IO.File]::WriteAllLines($resultPath, [string[]]@("request $($next.Name)", "RESULT: REJECTED ($reason)"), $utf8)
    } else {
      $cycleArgs = @("-NoProfile", "-File", $cycle, "-Action", $action, "-ResultFile", $resultPath)
      if ($action -eq "land") { $cycleArgs += @("-Landing", $landing) }
      if ($lock) { $cycleArgs += "-Lock" }
      if (-not $fix) { $cycleArgs += "-NoFix" }
      & powershell @cycleArgs
      Write-Host "=== request $($next.Name) finished with exit code $LASTEXITCODE; result in $resultPath"
    }
    Move-Item -Force -Path $next.FullName -Destination $processed
  }
  Start-Sleep -Seconds 5
}
