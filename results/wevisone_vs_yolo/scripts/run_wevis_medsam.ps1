# After the Automagic3 follow-up queue frees the GPU: WeVisOne boxes (+gate) -> fine-tuned MedSAM masks, 3 seeds.
param([int]$WaitPid = 0)
$ErrorActionPreference = "Continue"  # native stderr (progress bars) must not abort; every call checks $LASTEXITCODE
Set-Location $PSScriptRoot
$py = "C:/Users/luis/miniconda3/envs/wevisone/python.exe"
$env:PYTHONIOENCODING = "utf-8"
function Log($m) { "$(Get-Date -Format s) $m" | Tee-Object -Append ../runs/wevis_medsam.status }
if ($WaitPid) { while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep 60 } }
Log "start"
$modes = (42, 43, 44 | ForEach-Object { "wevis:../runs/wevis/baseline_s$_" }) -join ","
& $py medsam_eval.py --decoder ../runs/medsam_ft/best_decoder.pt --out ../runs/medsam_ft --modes $modes `
    --sets "busi:test,uclm:all,busuc:all,busbra:test,breast:all" *> ../runs/medsam_ft/_log_wevis_medsam.txt
if ($LASTEXITCODE) { Log "medsam_eval failed"; throw "medsam_eval" }
& $py wevis_medsam.py *> ../runs/comparison/_log_wevis_medsam.txt
if ($LASTEXITCODE) { Log "summary failed"; throw "wevis_medsam" }
Log "DONE"
"DONE" | Out-File ../runs/wevis_medsam.done
