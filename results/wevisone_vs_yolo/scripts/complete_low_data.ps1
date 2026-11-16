# Finish the authorized low-data study, audit it, export it, and optionally push the exact report files.
param([int]$WaitForQueuePid = 0, [switch]$Push)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "C:/Users/luis/miniconda3/envs/wevisone/python.exe"
$git = "C:/Program Files/Git/cmd/git.exe"
$repo = (Resolve-Path Medical-image-NTUT).Path
$statusFile = Join-Path (Get-Location) "runs/comparison/publish_status.json"
$env:PYTHONUTF8 = "1"
$env:GIT_TERMINAL_PROMPT = "0"
function Status($state, $detail) {
    @{state=$state; detail=$detail; updated=(Get-Date -Format s)} | ConvertTo-Json | Set-Content -Encoding UTF8 $statusFile
    "$(Get-Date -Format s) $state $detail"
}
function Check($step) { if ($LASTEXITCODE -ne 0) { throw "$step failed with exit code $LASTEXITCODE" } }
function Logged-Python($script, $stem) {
    $logBase = Join-Path (Get-Location) "runs/comparison/$stem"
    $taskProcess = Start-Process -FilePath $py -ArgumentList @($script) -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$logBase.out" -RedirectStandardError "$logBase.err"
    if ($taskProcess.ExitCode -ne 0) { throw "$script failed with exit code $($taskProcess.ExitCode); see $logBase.err" }
}
try {
    Status "waiting" "Waiting for all nine audited low-data runs; push requested: $Push"
    if ($WaitForQueuePid) {
        $running = Get-Process -Id $WaitForQueuePid -ErrorAction SilentlyContinue
        if ($running) { Wait-Process -Id $WaitForQueuePid }
        if (-not (Test-Path runs/queue_low_data.done)) { Status "verifying" "Queue exited without marker; checking whether training completed before a reporting failure" }
    } else {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/queue_master.ps1 -LowDataOnly
        Check "training queue"
    }
    Status "verifying" "Checking completed training and regenerating reports"
    & $py scripts/low_data_status.py
    Check "completion audit"
    & $py -m unittest discover -s scripts -p test_localization.py
    Check "localization tests"
    & $py -m unittest discover -s scripts -p test_fpr.py
    Check "FPR tests"
    Logged-Python scripts/analyze.py _log_analyze_final
    Logged-Python scripts/f1_table.py _log_f1_final
    & $py -m unittest discover -s scripts -p test_study_export.py
    Check "exported narrative tests"
    & $py scripts/fig_operating.py
    Check "operating figure"
    $dirty = & $git -C $repo status --porcelain
    Check "repository status"
    if ($dirty) { throw "Repository has local changes; preserving them and stopping before export" }
    $branch = & $git -C $repo branch --show-current
    if ($branch -ne "main") { throw "Expected main branch, got $branch" }
    $remote = & $git -C $repo remote get-url origin
    if ($remote -notin @("https://github.com/ShiQu1218/Medical-image-NTUT.git", "https://github.com/ShiQu1218/Medical-image-NTUT")) {
        throw "Unexpected Git origin; stopping before export"
    }
    if ($Push) {
        & $git -C $repo fetch origin
        Check "fetch"
        & $git -C $repo merge --ff-only origin/main
        Check "fast-forward"
    }
    & $py scripts/sync_study.py
    Check "audited report export"
    $manifest = @(Get-Content runs/comparison/publish_manifest.json -Raw | ConvertFrom-Json)
    & $git -C $repo diff --check -- @manifest
    Check "diff whitespace check"
    if (-not $Push) { Status "exported" "Audited results exported; no commit or push requested"; exit 0 }
    & $git -C $repo add -- @manifest
    Check "stage report manifest"
    $authorName = & $git -C $repo log -1 --format=%an
    $authorEmail = & $git -C $repo log -1 --format=%ae
    & $git -C $repo -c "user.name=$authorName" -c "user.email=$authorEmail" commit -m "Separate fixed-threshold inference, add strict localization, complete low-data seeds"
    Check "commit"
    & $git -C $repo push origin HEAD:main
    Check "push"
    $commit = & $git -C $repo rev-parse HEAD
    $published = & $git -C $repo ls-remote origin refs/heads/main
    Check "verify remote commit"
    if (-not $published.StartsWith($commit)) { throw "Remote main does not match committed results" }
    Status "pushed" "https://github.com/ShiQu1218/Medical-image-NTUT/commit/$commit"
} catch {
    Status "failed" $_.Exception.Message
    throw
}
