# 第二輪訓練參數

| 項目 | 設定 |
|---|---|
| 模型 | 從零訓練 U-Net，channels 16/32/64/128/256，1,940,817 個參數；無預訓練、BatchNorm、Dropout |
| 輸入 | 256×256 單通道灰階，uint8 前處理後除以 255 |
| 最佳化器 | Adam，lr=0.001，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 |
| 損失 | BCEWithLogitsLoss + (1 − 每張 soft Dice 的平均)，權重各 1；Dice smoothing=1 |
| 批次 | batch size=16，保留不足 16 張的末批次 |
| 更新預算 | 每組恰為 2,800 次；另保存第 1,400 次的最佳驗證模型 |
| 等效 epoch | 原版每 epoch 35 次更新：約 40／80 epochs；清理版每 epoch 23 次：約 60.87／121.74 epochs |
| 驗證 | 每 35 次更新；固定同一份 85 張乾淨驗證影像，選 hard Dice 最高模型，閾值 0.5 |
| 訓練集 | 原版 550 張；清理版 368 張；延用第一輪分組切分，切分種子 2026 |
| 訓練種子 | 42、43、44；每個資料版本／方法獨立從頭初始化，兩個預算來自同一條訓練軌跡 |
| 資料增強 | 僅訓練；旋轉 ±10°，縮放參數 0.95–1.05，零填補；影像雙線性、遮罩最近鄰，同步幾何轉換 |
| 前處理 | raw；CLAHE clip=2、grid=8×8；SRAD 20 iterations、step=0.5、全圖 q0；median 3×3；CLAHE→SRAD |
| 學習率排程／早停 | 無；固定更新預算以控制比較條件 |
| 精度／決定性 | FP32、channels_last，PyTorch 決定性演算法，cuDNN benchmark=False |
| 硬體／套件 | RTX 5080；Python 3.13、torch 2.11.0+cu128；其餘沿用第一輪 environment.txt |
| 測試集 | 本輪不載入、不前處理、不推論；報告只有驗證分數 |

重現：從 repo 根目錄執行 `.venv/Scripts/python.exe -u followup.py --output results/followup_reproduction`。
第一輪原始報告不覆寫。新的設定與程式雜湊保存在 config.json。

本輪只診斷更新預算與學習進度；尚未執行逐項清理消融、資料量匹配抽樣或前處理參數搜尋。
