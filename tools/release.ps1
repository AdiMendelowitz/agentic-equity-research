<#
.SYNOPSIS
  Publishes the tracked tree of this private repository as one commit in the public repository. Run by Adi in person.
.DESCRIPTION
  Dry run (default):  powershell -File tools\release.ps1
  Publish:            powershell -File tools\release.ps1 -Publish
  The private repository (agentic-equity-research-dev) keeps the full history and never becomes public. This script
  exports the files tracked at HEAD into a folder under %TEMP%, checks the export (no internal paths, gitleaks,
  redaction scan, fresh-environment quick start), and prints the file list. With -Publish it then asks for the word
  publish, places the export as a single new commit in the public repository and waits for its CI run.
  The public repository receives files only: no commit, branch, tag or Actions log of the private one.
  Every line is written, UTF-8, to notes\runs\release-<timestamp>.log.
#>
param(
  [switch]$Publish,
  [string]$PublicUrl = "https://github.com/AdiMendelowitz/agentic-equity-research.git",
  [string]$Message = "agentic-equity-research v0.1"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -Path $repo
$runs = Join-Path $repo "notes\runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $runs "release-$stamp.log"
$utf8 = New-Object System.Text.UTF8Encoding $false
$work = Join-Path $env:TEMP "aer-release-$stamp"
$export = Join-Path $work "export"
$public = Join-Path $work "public"
$zip = Join-Path $work "tree.zip"

function Write-Log([string[]]$lines) {
  foreach ($l in $lines) { Write-Host $l }
  [System.IO.File]::AppendAllLines($log, [string[]]$lines, $utf8)
}
function Stop-Release([string]$why) {
  Write-Log @("", "RESULT: REFUSED: $why", "log: $log")
  Set-Location -Path $repo
  exit 1
}
function Invoke-Logged([string]$commandLine) {
  Write-Log @("> $commandLine")
  cmd /c "$commandLine 2>&1" | ForEach-Object { Write-Log @([string]$_) }
  return $LASTEXITCODE
}

if ($work -match "\s") { throw "the temp folder path contains a space, which this script does not quote: $work" }
New-Item -ItemType Directory -Force -Path $work | Out-Null
$mode = "DRY RUN"
if ($Publish) { $mode = "PUBLISH" }
Write-Log @("release started $stamp, mode $mode", "private folder $repo", "public target $PublicUrl", "work folder $work")

# 1. the private side must be the renamed repository, clean, and authored under the noreply identity
$origin = (git --no-pager remote get-url origin)
if ($origin -notmatch 'agentic-equity-research-dev(\.git)?$') { Stop-Release "origin is not the private -dev repository: $origin" }
if ($PublicUrl -match 'agentic-equity-research-dev') { Stop-Release "the public target points at the private repository" }
$dirty = @(git --no-pager status --porcelain)
if ($dirty.Count -ne 0) { Stop-Release "$($dirty.Count) uncommitted path(s); land or discard them first" }
$head = (git --no-pager rev-parse --short HEAD)
$email = (git --no-pager config --get user.email)
$name = (git --no-pager config --get user.name)
if ($email -notmatch '^\d+\+[A-Za-z0-9-]+@users\.noreply\.github\.com$') { Stop-Release "author email is not a GitHub noreply address: $email" }
Write-Log @("private origin $origin", "private commit $head", "author $name <$email>")

# 2. export exactly the tracked files of HEAD
git --no-pager archive --format=zip --output $zip HEAD
if ($LASTEXITCODE -ne 0) { Stop-Release "git archive failed" }
Expand-Archive -Path $zip -DestinationPath $export
$files = @(Get-ChildItem -Path $export -Recurse -File -Force | ForEach-Object { $_.FullName.Substring($export.Length + 1).Replace("\", "/") } | Sort-Object)
Write-Log (@("", "=== exported files: $($files.Count)") + $files)

# 3. checks on the export, which is what a reader receives
$internal = @($files | Where-Object { $_ -match '^(notes/|CLAUDE\.md$)' -or $_ -match '(^|/)\.fuse_hidden' -or $_ -match '\.(db|sqlite|pkl|joblib)$' })
if ($internal.Count -ne 0) { Stop-Release "internal or generated files in the export: $($internal -join ', ')" }
foreach ($required in @("README.md", "LICENSE", "pyproject.toml", "uv.lock", ".github/workflows/ci.yml")) {
  if ($files -notcontains $required) { Stop-Release "required file missing from the export: $required" }
}
Write-Log @("", "=== gitleaks on the export")
if (Get-Command gitleaks -ErrorAction SilentlyContinue) {
  if ((Invoke-Logged "gitleaks dir $export --no-banner --redact --verbose") -ne 0) { Stop-Release "gitleaks reported a finding in the export" }
} else { Stop-Release "gitleaks is not installed; a release is not scanned by CI before it is public" }
Write-Log @("", "=== redaction scan on the export")
if ((Invoke-Logged "python $export\tools\redact_check.py") -ne 0) { Stop-Release "redaction scan reported a hit in the export" }
Write-Log @("", "=== fresh-environment quick start on the export")
# built from a copy, so no build artefact can enter the export
$qs = Join-Path $work "quickstart-source"
Copy-Item -Path $export -Destination $qs -Recurse
if ((Invoke-Logged "powershell -NoProfile -File $repo\tools\quickstart.ps1 -Source $qs") -ne 0) { Stop-Release "quick start failed on the export" }

if (-not $Publish) {
  Write-Log @("", "RESULT: DRY RUN PASS ($($files.Count) files from private commit $head would be published as one commit)", "log: $log")
  exit 0
}

# 4. publish: one new commit holding the export
$answer = Read-Host "Publish $($files.Count) files from private commit $head to $PublicUrl ? type publish"
if ($answer -ne "publish") { Stop-Release "cancelled at the prompt; nothing was pushed" }
$remoteHeads = @(git --no-pager ls-remote --heads $PublicUrl)
if ($LASTEXITCODE -ne 0) { Stop-Release "the public repository cannot be reached; create it empty on GitHub first" }
if ($remoteHeads.Count -eq 0) {
  New-Item -ItemType Directory -Force -Path $public | Out-Null
  Set-Location -Path $public
  git --no-pager init --initial-branch main | Out-Null
  git --no-pager remote add origin $PublicUrl
  $expectedCommits = 1
} else {
  git --no-pager clone --quiet $PublicUrl $public
  if ($LASTEXITCODE -ne 0) { Stop-Release "clone of the public repository failed" }
  Set-Location -Path $public
  $expectedCommits = [int](git --no-pager rev-list --count HEAD) + 1
  Get-ChildItem -Path $public -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force
}
Copy-Item -Path (Join-Path $export "*") -Destination $public -Recurse -Force
git --no-pager config user.email $email
git --no-pager config user.name $name
git --no-pager add --all
$staged = @(git --no-pager diff --cached --name-only)
if ($staged.Count -eq 0) { Stop-Release "the public repository already holds this tree; nothing to publish" }
git --no-pager commit --quiet --message $Message
if ($LASTEXITCODE -ne 0) { Stop-Release "commit failed" }
$tracked = @(git --no-pager ls-files)
$count = [int](git --no-pager rev-list --count HEAD)
$authors = @(git --no-pager log --format="%ae" | Sort-Object -Unique)
Write-Log @("public commits after this release: $count (expected $expectedCommits)", "public tracked files: $($tracked.Count) (export: $($files.Count))", "public authors: $($authors -join ', ')")
if ($count -ne $expectedCommits) { Stop-Release "unexpected commit count; nothing was pushed" }
if ($tracked.Count -ne $files.Count) { Stop-Release "tracked file count differs from the export; nothing was pushed" }
foreach ($a in $authors) { if ($a -notmatch 'users\.noreply\.github\.com$') { Stop-Release "non-noreply author in the public history: $a" } }
git --no-pager push --set-upstream origin main
if ($LASTEXITCODE -ne 0) { Stop-Release "push failed" }
$sha = (git --no-pager rev-parse HEAD)
Write-Log @("pushed $sha to $PublicUrl")
$ciCode = 0
if (Get-Command gh -ErrorAction SilentlyContinue) {
  $runId = $null
  foreach ($attempt in 1..8) {
    Start-Sleep -Seconds 5
    $runId = gh run list --commit $sha --limit 1 --json databaseId --jq ".[0].databaseId"
    if ($runId) { break }
  }
  if ($runId) {
    Write-Log @("waiting for the public CI run $runId")
    gh run watch $runId --exit-status
    $ciCode = $LASTEXITCODE
    Write-Log @("public ci exit code: $ciCode (0 means green)")
  } else { Write-Log @("no CI run found yet for $sha") }
}
Set-Location -Path $repo
Write-Log @("", "RESULT: PUBLISHED $sha ($($files.Count) files, $count commit(s) in the public repository)", "log: $log")
exit $ciCode
