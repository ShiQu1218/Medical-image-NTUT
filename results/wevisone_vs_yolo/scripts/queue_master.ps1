# Serial GPU queue for the WeVisOne-vs-YOLO study. Every step is skip-if-done, so the script can be rerun.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "python"
$EXT = "busuc:all,busbra:test,breast:all"   # uclm:test is derived from uclm:all records (patient filter)
function Log($m) { "$(Get-Date -Format s) $m" | Tee-Object -Append ../runs/queue_master.status }

function Run-Yolo($name, $train, $val, $sets, $seed, $epochs = 150, $patience = 40) {
    $data = "../data_yolo/$name"; $run = "../runs/yolo/${name}_s$seed"
    if (Test-Path "$run/eval/report.json") { return }
    Log "yolo $name s$seed"
    & $py yolo_data.py --out $data --train $train --val $val | Out-Null
    if ($LASTEXITCODE) { throw "export $name" }
    if (-not (Test-Path "$run/weights/best.pt")) {
        & $py yolo_train.py --data "$data/data.yaml" --out $run --seed $seed --epochs $epochs --patience $patience
        if ($LASTEXITCODE) { throw "train $name $seed" }
    }
    & $py yolo_eval.py --weights "$run/weights/best.pt" --out "$run/eval" --sets $sets
    if ($LASTEXITCODE) { throw "eval $name $seed" }
}

function Marker($run, $sets, $modes) {
    $legacy = "$run/eval/DONE_$($modes -replace '[:/\\,]','_')_$($sets -replace '[:/\\,+@]','_')"
    if ($legacy.Length -lt 200) { return $legacy }   # keep markers already written by earlier runs
    $h = [BitConverter]::ToString([Security.Cryptography.SHA1]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes("$modes|$sets"))).Replace("-", "").Substring(0, 12)
    return "$run/eval/DONE_$h"
}
function Eval-Wevis($run, $sets, $modes = "generate") {
    $m = Marker $run $sets $modes
    if (Test-Path $m) { return }
    Log "wevis eval $run $($sets.Substring(0, [Math]::Min(120, $sets.Length))) $modes"
    & $py wevis_eval.py --lora "$run/best_lora" --decoder "$run/best_decoder.pt" --out "$run/eval" --sets $sets --modes $modes
    if ($LASTEXITCODE) { throw "wevis eval $run" }
    "$modes | $sets" | Out-File $m
}

function Run-Wevis($name, $sets, [string[]]$trainArgs) {
    $run = "../runs/wevis/$name"
    if (-not (Test-Path "$run/best_decoder.pt") -or -not (Test-Path "$run/TRAIN_DONE")) {
        Log "wevis train $name"
        & $py train.py --out $run @trainArgs
        if ($LASTEXITCODE) { throw "wevis train $name" }
        "" | Out-File "$run/TRAIN_DONE"
    }
    Eval-Wevis $run $sets
}

# 0) YOLO BUSI baselines (separate queue) must be finished
while (-not (Test-Path ../runs/yolo/busi_queue.done)) { Start-Sleep 30 }

# 1) WeVisOne baseline seeds: external sets (+ GT-box oracle on UCLM for seed 42 runs in its own job)
foreach ($s in 42, 43, 44) {
    $src = if ($s -eq 42) { "../runs/lora_decoder" } else { "../runs/sweep/A_seed$s" }
    $dst = "../runs/wevis/baseline_s$s"
    if (-not (Test-Path "$dst/best_decoder.pt")) { New-Item -ItemType Directory -Force $dst | Out-Null; Copy-Item -Recurse "$src/best_lora" $dst; Copy-Item "$src/best_decoder.pt" $dst }
    if ($s -ne 42) { Eval-Wevis $dst "busi:val,busi:test,uclm:all" }   # s42 was run separately (incl. gt_box) into $dst root
}
foreach ($s in 42, 43, 44) { Eval-Wevis "../runs/wevis/baseline_s$s" $EXT }
foreach ($s in 42, 43, 44) {
    $run = "../runs/yolo/busi_m_s$s"
    if (-not (Test-Path "$run/eval_ext/report.json")) {
        Log "yolo eval_ext s$s"
        & $py yolo_eval.py --weights "$run/weights/best.pt" --out "$run/eval_ext" --sets "busi:val,$EXT"
        if ($LASTEXITCODE) { throw "yolo eval_ext $s" }
    }
    if (-not (Test-Path "$run/eval_breast/report.json")) {
        Log "yolo eval_breast s$s"
        & $py yolo_eval.py --weights "$run/weights/best.pt" --out "$run/eval_breast" --sets "busi:val,breast:all"
        if ($LASTEXITCODE) { throw "yolo eval_breast $s" }
    }
}

# 1a) Reviewer items: original (zero-shot) WeVisOne on every cohort; "+None" prompt on all seeds/cohorts
$ALL = "busi:val,busi:test,uclm:all,$EXT"
if (-not (Test-Path ../runs/wevis/zeroshot/DONE)) {
    Log "wevis zero-shot all cohorts"
    & $py wevis_eval.py --out ../runs/wevis/zeroshot --sets $ALL --modes generate
    if ($LASTEXITCODE) { throw "zeroshot" }
    "" | Out-File ../runs/wevis/zeroshot/DONE
}
foreach ($s in 42, 43, 44) {
    $out = if ($s -eq 42) { "../runs/wevis/prompt_none_s42" } else { "../runs/wevis/prompt_none_s$s" }
    if (-not (Test-Path "$out/DONE")) {
        Log "wevis +None prompt s$s"
        & $py wevis_eval.py --lora "../runs/wevis/baseline_s$s/best_lora" --decoder "../runs/wevis/baseline_s$s/best_decoder.pt" --out $out --sets $ALL --modes generate --prompt_suffix "If no breast lesion is visible, return None."
        if ($LASTEXITCODE) { throw "prompt $s" }
        "" | Out-File "$out/DONE"
    }
}

# 1b) Authoritative baselines: nnU-Net v2 (Nat Methods 2021) on our exact split; MedSAM (Nat Commun 2024) box-prompted
$env:nnUNet_raw = (Resolve-Path ../nnunet/raw).Path; $env:nnUNet_preprocessed = (Resolve-Path ../nnunet/preprocessed).Path
$env:nnUNet_results = (Join-Path (Resolve-Path ..).Path "nnunet/results")
$nnbin = "$env:CONDA_PREFIX/Scripts"
function Run-NnUnet($did, $dname, $val, [string[]]$tests, $run) {
    $model = "$env:nnUNet_results/Dataset${did}_$dname/nnUNetTrainer_250epochs__nnUNetPlans__2d/fold_0/checkpoint_final.pth"
    if (-not (Test-Path $model)) {
        Log "nnunet train $did"
        & "$nnbin/nnUNetv2_train.exe" $did 2d 0 -tr nnUNetTrainer_250epochs --npz
        if ($LASTEXITCODE) { throw "nnunet train $did" }
    }
    foreach ($t in $tests) {
        $key = $t -replace ':', '_' -replace '\+', '-'
        $jsonl = "../runs/nnunet/$run/$($t -replace ':', '_')_generate.jsonl"
        if (Test-Path $jsonl) { continue }
        Log "nnunet predict $did $t"
        $pred = "../nnunet/pred/$run/$key"
        & "$nnbin/nnUNetv2_predict.exe" -i "../nnunet/raw/Dataset${did}_$dname/imagesTs_$key" -o $pred -d $did -c 2d -f 0 -tr nnUNetTrainer_250epochs --save_probabilities
        if ($LASTEXITCODE) { throw "nnunet predict $t" }
        & $py nnunet_eval.py --pred $pred --images "../nnunet/raw/Dataset${did}_$dname/imagesTs_$key" --out "../runs/nnunet/$run" --val_spec $val --delete_probs
        if ($LASTEXITCODE) { throw "nnunet eval $t" }
    }
}
Run-NnUnet 501 "BUSIclean" "busi:val" @("busi:val", "busi:test", "uclm:all", "busuc:all", "busbra:test", "breast:all") "busi_s0"
if (-not (Test-Path ../runs/medsam/DONE_gt)) {
    Log "medsam gt_box"
    & $py medsam_eval.py --sets "busi:val,busi:test,uclm:all,busuc:all,busbra:test,breast:all" --modes gt_box
    if ($LASTEXITCODE) { throw "medsam gt" }
    "" | Out-File ../runs/medsam/DONE_gt
}
foreach ($s in 42, 43, 44) {
    $thr = (Get-Content "../runs/yolo/busi_m_s$s/eval/report.json" -Raw | ConvertFrom-Json).calibrated_threshold
    if (-not (Test-Path "../runs/medsam/DONE_yolo_s$s")) {
        Log "medsam yolo boxes s$s"
        & $py medsam_eval.py --sets "busi:val,busi:test,uclm:all" --modes "yolo:../runs/yolo/busi_m_s$s/eval:$thr,yolo:../runs/yolo/busi_m_s$s/eval:0.25"
        if ($LASTEXITCODE) { throw "medsam yolo $s" }
        "" | Out-File "../runs/medsam/DONE_yolo_s$s"
    }
}

# 2) WeVisOne strong augmentation, 3 seeds, same 230-update budget
foreach ($s in 42, 43, 44) {
    Run-Wevis "aug_s$s" "busi:val,busi:test,uclm:all,$EXT" @("--seed", "$s", "--aug", "strong")
}

# 2b) Synthetic domain-shift stress test (corruptions.py; parameters fixed a priori)
$CORR = (@("gain", "speckle", "blur", "tgc", "jpeg", "icon", "sector", "fov") | ForEach-Object { $f = $_; 1, 2 | ForEach-Object { "busi:test~$f$_", "uclm:test~$f$_" } }) -join ","
foreach ($r in "baseline_s42", "aug_s42") { Eval-Wevis "../runs/wevis/$r" "busi:val,$CORR" }
if (-not (Test-Path "../runs/yolo/busi_m_s42/eval_corrupt/report.json")) {
    Log "yolo corruptions"
    & $py yolo_eval.py --weights "../runs/yolo/busi_m_s42/weights/best.pt" --out "../runs/yolo/busi_m_s42/eval_corrupt" --sets "busi:val,$CORR"
    if ($LASTEXITCODE) { throw "yolo corrupt" }
}
& $py nnunet_prep.py --ids 501 --extra_tests $CORR
if ($LASTEXITCODE) { throw "nnunet corrupt prep" }
Run-NnUnet 501 "BUSIclean" "busi:val" ($CORR -split ",") "busi_s0"

# 2c) Memorization probe (corruptions.py hflip/rot/zoom/shift): BUS_UC (suspected pretraining exposure) vs controls
$MEMO = (@("hflip1", "rot1", "zoom1", "shift1") | ForEach-Object { $c = $_; "busi:test", "uclm:all", "busuc:all", "busbra:test", "breast:all" | ForEach-Object { "$_~$c" } }) -join ","
if (-not (Test-Path ../runs/wevis/zeroshot_memo/DONE)) {
    Log "wevis zero-shot memorization probe"
    & $py wevis_eval.py --out ../runs/wevis/zeroshot_memo --sets "busi:val,$MEMO" --modes generate
    if ($LASTEXITCODE) { throw "zeroshot memo" }
    "" | Out-File ../runs/wevis/zeroshot_memo/DONE
}
Eval-Wevis "../runs/wevis/baseline_s42" "busi:val,$MEMO"
& $py nnunet_prep.py --ids 501 --extra_tests $MEMO
if ($LASTEXITCODE) { throw "nnunet memo prep" }
Run-NnUnet 501 "BUSIclean" "busi:val" ($MEMO -split ",") "busi_s0"

# 3) Grouped 5-fold CV over all clean BUSI (baseline config for both models; YOLO 1 seed, WeVisOne 1 seed)
foreach ($k in 0..4) {
    Run-Yolo "cv$k" "busicv${k}:train" "busicv${k}:val" "busicv${k}:val,busicv${k}:test" 42
    Run-Wevis "cv${k}_s42" "busicv${k}:val,busicv${k}:test" @("--train_spec", "busicv${k}:train", "--val_spec", "busicv${k}:val", "--total_steps", "230")
}

# 4) Low-data regime: subsets nested per seed; WeVisOne keeps the 230-update budget, validates 5 times
foreach ($f in "0.1", "0.25", "0.5") {
    foreach ($s in 42, 43, 44) { Run-Yolo "frac${f}_sub$s" "busi:train@$f/$s" "busi:val" "busi:val,busi:test,uclm:all" $s }
    Run-Wevis "frac${f}_s42" "busi:val,busi:test,uclm:all" @("--train_spec", "busi:train@$f/42", "--total_steps", "230", "--n_vals", "5")
}

# 5) Multi-source (strong aug for WeVisOne, compare against aug_s*): (b) UCLM held out, (c) + UCLM train patients
$msb = @("busi:train+busbra:train", "busi:val+busbra:val")
$msc = @("busi:train+busbra:train+uclm:train", "busi:val+busbra:val+uclm:val")
foreach ($s in 42, 43) {
    Run-Yolo "ms_bra" $msb[0] $msb[1] "$($msb[1]),busi:test,uclm:all,$EXT" $s 80 25
    Run-Yolo "ms_bra_uclm" $msc[0] $msc[1] "$($msc[1]),busi:test,uclm:all,$EXT" $s 80 25
}
Run-Wevis "ms_bra_s42" "$($msb[1]),busi:test,uclm:all,$EXT" @("--train_spec", $msb[0], "--val_spec", $msb[1], "--aug", "strong", "--total_steps", "690", "--n_vals", "4")
Run-Wevis "ms_bra_uclm_s42" "$($msc[1]),busi:test,uclm:all,$EXT" @("--train_spec", $msc[0], "--val_spec", $msc[1], "--aug", "strong", "--total_steps", "690", "--n_vals", "4")

# 6) Hybrid: YOLO boxes (val-calibrated threshold) -> WeVisOne baseline decoder
foreach ($s in 42, 43, 44) {
    $rep = Get-Content "../runs/yolo/busi_m_s$s/eval/report.json" -Raw | ConvertFrom-Json
    $thr = $rep.calibrated_threshold
    foreach ($t in @($thr, 0.25)) {
        Eval-Wevis "../runs/wevis/baseline_s$s" "busi:test,uclm:all" "yolo:../runs/yolo/busi_m_s$s/eval:$t"
        Eval-Wevis "../runs/wevis/baseline_s$s" "busi:val" "yolo:../runs/yolo/busi_m_s$s/eval:$t"
    }
}
# 7) nnU-Net trained on BUS-BRA, tested on all 647 BUSI lesions (replicates the CMT-BUSNet nnU-Net transfer setting) + UCLM
Run-NnUnet 502 "BUSBRA" "busbra:val" @("busbra:val", "busbra:test", "busirand0:train+busirand0:val+busirand0:test", "uclm:all") "busbra_s0"

# 8) Multi-source extra seeds: YOLO s44, WeVisOne s43/s44 (same config as s42)
Run-Yolo "ms_bra" $msb[0] $msb[1] "$($msb[1]),busi:test,uclm:all,$EXT" 44 80 25
Run-Yolo "ms_bra_uclm" $msc[0] $msc[1] "$($msc[1]),busi:test,uclm:all,$EXT" 44 80 25
foreach ($s in 43, 44) {
    Run-Wevis "ms_bra_s$s" "$($msb[1]),busi:test,uclm:all,$EXT" @("--train_spec", $msb[0], "--val_spec", $msb[1], "--aug", "strong", "--total_steps", "690", "--n_vals", "4", "--seed", "$s")
    Run-Wevis "ms_bra_uclm_s$s" "$($msc[1]),busi:test,uclm:all,$EXT" @("--train_spec", $msc[0], "--val_spec", $msc[1], "--aug", "strong", "--total_steps", "690", "--n_vals", "4", "--seed", "$s")
}
Log "ALL DONE"
"DONE" | Out-File ../runs/queue_master.done
