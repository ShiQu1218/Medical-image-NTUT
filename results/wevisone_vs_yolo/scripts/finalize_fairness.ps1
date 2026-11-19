# Wait for queue_fairness + MedSAM fine-tune, then: clean YOLO latency benchmark, regenerate all reports.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "C:/Users/luis/miniconda3/envs/wevisone/python.exe"
$env:PYTHONIOENCODING = "utf-8"
function Log($m) { "$(Get-Date -Format s) $m" | Tee-Object -Append ../runs/queue_fairness.status }
while (-not ((Test-Path ../runs/queue_fairness.done) -and (Test-Path ../runs/medsam_ft/ALL_DONE))) { Start-Sleep 60 }
Log "finalize: benchmarks (GPU idle)"
foreach ($m in "yolo", "nnunet", "wevis") { & $py bench_cost.py --model $m *> "../runs/comparison/_log_bench_$m.txt"; if ($LASTEXITCODE) { throw "bench $m" } }
foreach ($s in "analyze", "stratify", "f1_table") {
    Log "finalize: $s"
    & $py "$s.py" *> "../runs/comparison/_log_final_$s.txt"
    if ($LASTEXITCODE) { throw "$s" }
}
Log "FINALIZE DONE"
"DONE" | Out-File ../runs/finalize_fairness.done
