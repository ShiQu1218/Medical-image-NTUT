# Fine-tuned MedSAM decoder: train, then GT-box and YOLO-cal-box evaluation (skip-if-done).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "C:/Users/luis/miniconda3/envs/wevisone/python.exe"
$out = "../runs/medsam_ft"
if (-not (Test-Path "$out/TRAIN_DONE")) { & $py medsam_finetune.py; if ($LASTEXITCODE) { throw "medsam ft train" } }
if (-not (Test-Path "$out/DONE_gt")) {
    & $py medsam_eval.py --decoder "$out/best_decoder.pt" --out $out --sets "busi:val,busi:test,uclm:all,busuc:all,busbra:test,breast:all" --modes gt_box
    if ($LASTEXITCODE) { throw "medsam ft gt" }; "" | Out-File "$out/DONE_gt"
}
foreach ($s in 42, 43, 44) {
    if (Test-Path "$out/DONE_yolo_s$s") { continue }
    $thr = (Get-Content "../runs/yolo/busi_m_s$s/eval/report.json" -Raw | ConvertFrom-Json).calibrated_threshold
    & $py medsam_eval.py --decoder "$out/best_decoder.pt" --out $out --sets "busi:test,uclm:all" --modes "yolo:../runs/yolo/busi_m_s$s/eval:$thr"
    if ($LASTEXITCODE) { throw "medsam ft yolo $s" }; "" | Out-File "$out/DONE_yolo_s$s"
}
"DONE" | Out-File "$out/ALL_DONE"
