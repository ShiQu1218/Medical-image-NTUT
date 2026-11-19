# Fairness follow-ups (2026-11-19): YOLO tuning effort parity and nnU-Net seeds. Serial, skip-if-done, rerunnable.
#  A) YOLO26m-seg + ultrasound strong aug (us_aug.py + degrees=10; matches WeVisOne --aug strong), seeds 42/43/44
#  B) YOLO26x-seg (largest size), default recipe, seeds 42/43/44
#  C) nnU-Net v2 2d: two more trainings on the SAME train/val split (splits_final folds 1,2 = copies of fold 0),
#     i.e. only the random initialisation / sampling differs -> runs/nnunet/busi_s1, busi_s2
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "C:/Users/luis/miniconda3/envs/wevisone/python.exe"
$ALLY = "busi:val,busi:test,uclm:all,busuc:all,busbra:test,breast:all"   # busi:val first = calibration split
function Log($m) { "$(Get-Date -Format s) $m" | Tee-Object -Append ../runs/queue_fairness.status }

function Run-YoloVariant($run, $model, $aug, $seed) {
    if (Test-Path "$run/eval/report.json") { return }
    if (-not (Test-Path "$run/weights/best.pt") -or -not (Test-Path "$run/TRAIN_DONE")) {
        Log "yolo train $run ($model, aug=$aug)"
        & $py yolo_train.py --data "../data_yolo/busi/data.yaml" --model $model --out $run --seed $seed --aug $aug
        if ($LASTEXITCODE) { throw "train $run" }
        "" | Out-File "$run/TRAIN_DONE"
    }
    Log "yolo eval $run"
    & $py yolo_eval.py --weights "$run/weights/best.pt" --out "$run/eval" --sets $ALLY
    if ($LASTEXITCODE) { throw "eval $run" }
}

Log "FAIRNESS QUEUE START"
foreach ($s in 42, 43, 44) { Run-YoloVariant "../runs/yolo/busi_maug_s$s" "yolo26m-seg.pt" "strong" $s }
foreach ($s in 42, 43, 44) { Run-YoloVariant "../runs/yolo/busi_x_s$s" "yolo26x-seg.pt" "default" $s }

$env:nnUNet_raw = (Resolve-Path ../nnunet/raw).Path; $env:nnUNet_preprocessed = (Resolve-Path ../nnunet/preprocessed).Path
$env:nnUNet_results = (Join-Path (Resolve-Path ..).Path "nnunet/results")
$nnbin = "C:/Users/luis/miniconda3/envs/wevisone/Scripts"
& $py -c "import json,shutil,pathlib; p=pathlib.Path(r'$env:nnUNet_preprocessed')/'Dataset501_BUSIclean'/'splits_final.json'; s=json.loads(p.read_text()); b=p.with_name('splits_final.fold0_only.json'); b.exists() or shutil.copy(p,b); s=[s[0]]*3 if len(s)==1 else s; assert all(f==s[0] for f in s); p.write_text(json.dumps(s,indent=2))"
if ($LASTEXITCODE) { throw "nnunet splits" }
$base = "$env:nnUNet_results/Dataset501_BUSIclean/nnUNetTrainer_250epochs__nnUNetPlans__2d"
foreach ($f in 1, 2) {
    if (-not (Test-Path "$base/fold_$f/checkpoint_final.pth")) {
        Log "nnunet train 501 fold $f (copy of fold 0 split)"
        & "$nnbin/nnUNetv2_train.exe" 501 2d $f -tr nnUNetTrainer_250epochs --npz
        if ($LASTEXITCODE) { throw "nnunet train fold $f" }
    }
    foreach ($t in $ALLY -split ",") {
        $key = $t -replace ':', '_'
        $jsonl = "../runs/nnunet/busi_s$f/${key}_generate.jsonl"
        if (Test-Path $jsonl) { continue }
        Log "nnunet predict fold $f $t"
        $pred = "../nnunet/pred/busi_s$f/$key"
        & "$nnbin/nnUNetv2_predict.exe" -i "../nnunet/raw/Dataset501_BUSIclean/imagesTs_$key" -o $pred -d 501 -c 2d -f $f -tr nnUNetTrainer_250epochs --save_probabilities
        if ($LASTEXITCODE) { throw "nnunet predict $f $t" }
        & $py nnunet_eval.py --pred $pred --images "../nnunet/raw/Dataset501_BUSIclean/imagesTs_$key" --out "../runs/nnunet/busi_s$f" --val_spec "busi:val" --delete_probs
        if ($LASTEXITCODE) { throw "nnunet eval $f $t" }
    }
}
Log "FAIRNESS QUEUE DONE"
"DONE" | Out-File ../runs/queue_fairness.done
