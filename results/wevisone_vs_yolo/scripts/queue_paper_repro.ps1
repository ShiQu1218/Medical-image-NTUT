# Replicate Yu et al. 2026 (QIMS) YOLO protocol: uncleaned BUSI lesions, image-level random 80/10/10,
# yolo11n-seg, 512 px, 50 epochs (no early stop), Ultralytics default aug. Waits for the prompt eval (GPU).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "python"
while (Get-Process -Id 4228 -ErrorAction SilentlyContinue) { Start-Sleep 20 }
foreach ($s in 0, 1, 2) {
    $name = "paper_busirand$s"; $data = "../data_yolo/$name"; $run = "../runs/yolo/${name}_y11n"
    & $py yolo_data.py --out $data --train "busirand${s}:train" --val "busirand${s}:val"
    if ($LASTEXITCODE) { throw "export $s" }
    if (-not (Test-Path "$run/weights/best.pt")) {
        & $py yolo_train.py --data "$data/data.yaml" --model yolo11n-seg.pt --out $run --seed $s --epochs 50 --patience 100 --imgsz 512
        if ($LASTEXITCODE) { throw "train $s" }
    }
    if (-not (Test-Path "$run/eval/report.json")) {
        & $py yolo_eval.py --weights "$run/weights/best.pt" --out "$run/eval" --imgsz 512 --sets "busirand${s}:val,busirand${s}:test,uclm:all"
        if ($LASTEXITCODE) { throw "eval $s" }
    }
}
"DONE" | Out-File ../runs/yolo/paper_repro.done
