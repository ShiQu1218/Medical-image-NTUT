# YOLO26m-seg on cleaned BUSI, 3 seeds; each evaluated on BUSI val/test + BUS-UCLM.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "C:/Users/luis/miniconda3/envs/wevisone/python.exe"
foreach ($seed in 42, 43, 44) {
    $run = "../runs/yolo/busi_m_s$seed"
    if (-not (Test-Path "$run/weights/best.pt")) {
        & $py yolo_train.py --data ../data_yolo/busi/data.yaml --out $run --seed $seed
        if ($LASTEXITCODE) { throw "train $seed failed" }
    }
    if (-not (Test-Path "$run/eval/report.json")) {
        & $py yolo_eval.py --weights "$run/weights/best.pt" --out "$run/eval"
        if ($LASTEXITCODE) { throw "eval $seed failed" }
    }
}
"DONE" | Out-File ../runs/yolo/busi_queue.done
