# Follow-up WeVisOne runs (2026-11-18). Serial, skip-if-done, rerunnable. Starts after queue_fairness finishes (GPU).
#  A) Longer training, LoRA + AdamW + cosine (schedule stretched, not continued): 460 and 690 updates, seeds 42/43/44,
#     5 evenly spaced validations like the 230-update baseline.
#  B) LoKr + AdamW + cosine, 230 updates: start-LR chosen on busi:val only (seed 42: 1e-4 / 3e-4 / 1e-3, best val Dice),
#     then seeds 43/44 at that LR. Decoder unchanged (AdamW 3e-5 cosine).
#  C) LoKr + Automagic3 (ostris ai-toolkit, vendored in third_party/) + constant, start LR 1e-4 for both groups,
#     230 updates, seeds 42/43/44. Only with -IncludeAutomagic3 (needs the user's approval to run vendored code).
param([switch]$IncludeAutomagic3)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "C:/Users/luis/miniconda3/envs/wevisone/python.exe"
$env:PYTHONIOENCODING = "utf-8"
$ALL = "busi:val,busi:test,uclm:all,busuc:all,busbra:test,breast:all"
function Log($m) { "$(Get-Date -Format s) $m" | Tee-Object -Append ../runs/queue_followup.status }

function Train-Wevis($name, [string[]]$trainArgs) {
    $run = "../runs/wevis/$name"
    if ((Test-Path "$run/best_decoder.pt") -and (Test-Path "$run/TRAIN_DONE")) { return }
    Log "wevis train $name $($trainArgs -join ' ')"
    & $py train.py --out $run @trainArgs
    if ($LASTEXITCODE) { throw "wevis train $name" }
    "" | Out-File "$run/TRAIN_DONE"
}
function Eval-Wevis($name) {
    $run = "../runs/wevis/$name"
    if (Test-Path "$run/eval/DONE_ALL6") { return }
    Log "wevis eval $name"
    & $py wevis_eval.py --lora "$run/best_lora" --decoder "$run/best_decoder.pt" --out "$run/eval" --sets $ALL --modes generate
    if ($LASTEXITCODE) { throw "wevis eval $name" }
    "" | Out-File "$run/eval/DONE_ALL6"
}

while (-not (Test-Path ../runs/queue_fairness.done)) { Start-Sleep 60 }
Log "FOLLOWUP QUEUE START (automagic3=$IncludeAutomagic3)"

# A) longer schedules
foreach ($steps in 460, 690) {
    foreach ($s in 42, 43, 44) {
        $n = "long${steps}_s$s"
        Train-Wevis $n @("--seed", "$s", "--total_steps", "$steps", "--n_vals", "5")
        Eval-Wevis $n
    }
}

# B) LoKr + AdamW: LR picked on val (seed 42), then seeds 43/44
foreach ($lr in "1e-4", "3e-4", "1e-3") { Train-Wevis "lokr_lr${lr}_s42" @("--adapter", "lokr", "--lr_lora", $lr, "--seed", "42", "--total_steps", "230", "--n_vals", "5") }
$best = & $py -c "import json,pathlib; r={lr: json.loads(pathlib.Path(f'../runs/wevis/lokr_lr{lr}_s42/best.json').read_text())['dice'] for lr in ('1e-4','3e-4','1e-3')}; print(max(r, key=r.get))"
if ($LASTEXITCODE) { throw "lokr lr select" }
$best = $best.Trim()
Log "LoKr AdamW: val-selected start LR = $best"
"$best" | Out-File ../runs/wevis/lokr_selected_lr.txt
$src = "../runs/wevis/lokr_lr${best}_s42"; $dst = "../runs/wevis/lokr_s42"
if (-not (Test-Path "$dst/TRAIN_DONE")) { New-Item -ItemType Directory -Force $dst | Out-Null; Copy-Item -Recurse -Force "$src/*" $dst }
foreach ($s in 43, 44) { Train-Wevis "lokr_s$s" @("--adapter", "lokr", "--lr_lora", $best, "--seed", "$s", "--total_steps", "230", "--n_vals", "5") }
foreach ($s in 42, 43, 44) { Eval-Wevis "lokr_s$s" }

# C) LoKr + Automagic3 + constant
if ($IncludeAutomagic3) {
    foreach ($s in 42, 43, 44) {
        $n = "lokr_am3_s$s"
        Train-Wevis $n @("--adapter", "lokr", "--optimizer", "automagic3", "--schedule", "constant", "--lr_lora", "1e-4", "--lr_decoder", "1e-4", "--seed", "$s", "--total_steps", "230", "--n_vals", "5")
        Eval-Wevis $n
    }
}

Log "followup: regenerate reports"
foreach ($sc in "analyze", "stratify", "f1_table") { & $py "$sc.py" *> "../runs/comparison/_log_followup_$sc.txt"; if ($LASTEXITCODE) { throw $sc } }
Log "FOLLOWUP QUEUE DONE"
"DONE" | Out-File ../runs/queue_followup.done
