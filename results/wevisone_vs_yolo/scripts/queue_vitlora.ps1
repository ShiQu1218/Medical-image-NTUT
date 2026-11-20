# LoRA on LLM + ViT blocks (2026-11-19). Same recipe as baseline_s* (LoRA r16/a32, AdamW 1e-4 / decoder 3e-5,
# cosine, 230 updates, 5 vals); seg loss not detached so it reaches the ViT LoRA. Waits for run_wevis_medsam.
$ErrorActionPreference = "Continue"  # native stderr (progress bars) must not abort; every call checks $LASTEXITCODE
Set-Location $PSScriptRoot
$py = "C:/Users/luis/miniconda3/envs/wevisone/python.exe"
$env:PYTHONIOENCODING = "utf-8"
$ALL = "busi:val,busi:test,uclm:all,busuc:all,busbra:test,breast:all"
function Log($m) { "$(Get-Date -Format s) $m" | Tee-Object -Append ../runs/queue_vitlora.status }
while (-not (Test-Path ../runs/wevis_medsam.done)) { Start-Sleep 60 }
Log "start"
$smoke = "../runs/wevis/_smoke_vitlora"
if (-not (Test-Path "$smoke/best_decoder.pt")) {
    Log "smoke test"
    & $py train.py --out $smoke --vit_lora --max_steps 4 --accum 2 --val_limit 3 *> ../runs/wevis/_smoke_vitlora.log
    if ($LASTEXITCODE) { Log "smoke FAILED (see runs/wevis/_smoke_vitlora.log)"; throw "smoke" }
}
foreach ($s in 42, 43, 44) {
    $run = "../runs/wevis/vitlora_s$s"
    if (-not (Test-Path "$run/TRAIN_DONE")) {
        Log "train vitlora_s$s"
        & $py train.py --out $run --vit_lora --seed $s --total_steps 230 --n_vals 5
        if ($LASTEXITCODE) { throw "train vitlora_s$s" }
        "" | Out-File "$run/TRAIN_DONE"
    }
    if (-not (Test-Path "$run/eval/DONE_ALL6")) {
        Log "eval vitlora_s$s"
        & $py wevis_eval.py --lora "$run/best_lora" --decoder "$run/best_decoder.pt" --out "$run/eval" --sets $ALL --modes generate
        if ($LASTEXITCODE) { throw "eval vitlora_s$s" }
        "" | Out-File "$run/eval/DONE_ALL6"
    }
}
Log "regenerate reports"
foreach ($sc in "analyze", "stratify", "f1_table") { & $py "$sc.py" *> "../runs/comparison/_log_vitlora_$sc.txt"; if ($LASTEXITCODE) { throw $sc } }
Log "DONE"
"DONE" | Out-File ../runs/queue_vitlora.done
