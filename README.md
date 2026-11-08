# 影像前處理對乳房超音波腫瘤分割的影響

G07 專題實驗：固定 U-Net，比較 BUSI 原版與清理版的五種前處理。
提案 PDF 是研究背景；本次新增中值濾波、CLAHE→SRAD，以及原版／清理版比較。

最新的**等樣本數品質比較**已完成：兩組皆訓練 368 張，按類別匹配，各跑三個種子與五種前處理；共同保存 1,400／2,800 次更新的結果。
本輪只有驗證集評估，與下方第一輪測試分數分開解讀。見 [第二輪報告](results/quality_matched/report.md)、[分析](results/quality_matched/analysis.md) 與 [完整訓練參數](results/quality_matched/parameters.md)。十一項測試通過。

重現第二輪：`.venv/Scripts/python.exe -u followup.py --matched --output results/quality_matched_reproduction`。

以下為第一輪：30 次實際訓練（每次 40 epochs、三個種子）。
下表是在**相同 76 張乾淨測試影像**上的平均 Dice：

| 前處理 | BUSI 原版 780 張 | 清理版 529 張 |
|---|---:|---:|
| 原圖 | 0.5657 | 0.6250 |
| CLAHE | 0.5974 | 0.4971 |
| SRAD | 0.6057 | 0.4474 |
| 中值濾波 | **0.6351** | 0.6058 |
| CLAHE→SRAD | 0.5337 | 0.4879 |

本次原版以中值濾波最高，清理版以原圖最高；CLAHE→SRAD 未改善表現。
28／30 次的最佳 epoch 位於最後六個 epoch，尚未確認訓練收斂；排名只適用於本次固定預算。
完整變異、案例與限制見 [實驗報告](results/busi/report.md) 與 [結果分析](results/busi/analysis.md)。

## 實驗設計

| 項目 | 設定 |
|---|---|
| 原版 | BUSI 全部 780 張，保留正常、良性、惡性影像 |
| 清理版 | 依已發表勘誤群組去重，排除腋下、切片針及類別衝突；529 張 |
| 切分 | 先以作者重複群組及完全相同像素合併，再按群組類別約 70/15/15；切分種子 2026 |
| 驗證 | 兩版共用乾淨驗證集，選 Dice 最高的 epoch |
| 主要測試 | 所有模型共用乾淨測試集，配對比較資料清理與前處理影響 |
| 補充測試 | 同時輸出原版模型在原版測試集、清理版模型在乾淨測試集的 native 指標 |
| 模型 | 從零訓練 U-Net；四次降採樣、起始 16 channels；無預訓練權重 |
| 訓練 | 256×256 灰階、batch 16、Adam lr 0.001、BCE + soft Dice、40 epochs |
| 重複訓練 | 種子 42、43、44；共 2×5×3 = 30 次 |
| 資料增強 | 只有訓練集；影像與遮罩同步旋轉 ±10°、縮放 0.95–1.05；不做上下翻轉 |
| 評估 | 0.5 固定閾值；逐張 Dice/IoU、病灶 Dice、正常影像偽陽性率 |

原版和清理版共用每張影像的分區。重複／近似影像不跨區；清理版是原版各分區的子集。
原版保留原始資料內容，但不是允許資料洩漏的逐張隨機切分。
兩版本使用相同訓練 epoch，樣本數及最佳化更新次數會不同。
BUSI 沒有可用病人 ID，分組不能保證病人獨立。

| 資料版本 | 訓練分區 | 驗證分區 | 測試分區 |
|---|---:|---:|---:|
| 原版 | 550 | 113 | 117 |
| 清理版 | 368 | 85 | 76 |

實際選模型時，兩版本都只使用上述 85 張乾淨驗證影像；主表都以 76 張乾淨測試影像評估。
模型初始化與訓練種子固定，啟用 PyTorch 決定性演算法；不同硬體與版本仍可能產生差異。

## 前處理

共用流程：讀取灰階、影像面積插值縮至 256×256、方法前處理、除以 255。
遮罩先合併同張影像的所有 `_mask*.png`，再以最近鄰縮放，保持二元標註。

| 方法 | 處理 |
|---|---|
| raw | 只有共用流程 |
| clahe | clip limit 2.0、8×8 tiles |
| srad | 20 iterations、step 0.5 |
| median | 3×3 中值濾波 |
| clahe_srad | 先 CLAHE，再 SRAD；沒有再加中值濾波 |

SRAD 依 Yu & Acton 的瞬時變異係數與導流係數，使用四鄰域、對稱面通量與無通量邊界；
導流係數限制 [0,1]，更新係數為 step/4。零值用 1/255 正位移處理，結束後扣回，不逐張重新拉伸。
q0 每輪從全圖變異／平均平方估計；未用人工均質 ROI，也未宣稱與論文所有實作細節完全相同。
參數預先固定，沒有看測試結果再調整。

## 重現

Windows PowerShell，Python 3.13 與 uv；CUDA 版使用本次已驗證的 PyTorch 版本：

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe prepare_data.py
.venv/Scripts/python.exe -m unittest -v
.venv/Scripts/python.exe -u experiment.py --output results/reproduction
```

CPU 可由一般 PyPI 安裝 requirements.txt，但完整訓練較慢。所有命令從 repo 根目錄執行。
已發表的結果放在 `results/busi/`；重現時另用 `results/reproduction/`，避免混用既有指標與未隨 repo 上傳的權重／預測。
資料下載採公開 Kaggle 鏡像，核對 ZIP SHA-256；來源變動時程式會停止，避免靜默換資料。
資料與模型權重留在本機，不納入 Git；提案、程式、作者勘誤清單、逐張結果與報告納入 repo。
GitHub 上的結果可直接閱讀，不需先下載 BUSI。

快速驗證完整流程，使用獨立輸出目錄避免混入正式結果：

```powershell
.venv/Scripts/python.exe experiment.py --epochs 1 --seeds 42 --size 64 --output results/smoke
```

相同設定重跑時略過已有 `metrics.json` 的完成訓練；未完成的單次訓練從頭重跑。
輸出目錄記錄程式與資料清單雜湊；設定或程式改變時必須另選 `--output`。

## 結果檔案

- [實驗報告](results/busi/report.md)：主要比較表、種子間變異、配對差異與成功／失敗案例。
- `results/busi/summary.csv`：每方法／資料版本的跨種子摘要。
- `results/busi/runs.csv`：每次訓練的最佳 epoch、時間、驗證與測試分數。
- `results/busi/per_image.csv`：各種子所有原版測試影像的分數，含乾淨子集標記。
- `results/busi/manifest.csv`：來源檔名、遮罩、重複群組、分區、像素雜湊及排除原因。
- `results/busi/config.json`、`dataset_stats.json`：實際環境、參數與類別數量。
- `results/busi/environment.txt`：本次執行的完整套件版本。
- `results/busi/checkpoints/`：各次訓練的每 epoch 紀錄與測試指標；最佳模型和第一種子的測試預測僅留本機。

空標註與空預測 Dice/IoU 設為 1；因此另外列出病灶影像 Dice，避免正常影像使總分看起來較高。
正常影像偽陽性率以「至少一個陽性預測像素」為單位。主要指標是影像平均，再跨訓練種子平均；
種子 SD 不是病人信賴區間。未做交叉驗證、外部測試或臨床驗證。

## 來源

- Al-Dhabyani et al. (2020), [Dataset of breast ultrasound images](https://doi.org/10.1016/j.dib.2019.104863)。
- [BUSI 原作者資料頁](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)；本次下載自 [Kaggle 公開鏡像](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)，不重新散布原始影像。
- Pawłowska et al. (2023), [勘誤補充清單](https://doi.org/10.17632/k8t3gnx9h6.1)，CC BY 4.0；詳見 [metadata](metadata/README.md)。
- Yu & Acton (2002), [Speckle reducing anisotropic diffusion](https://doi.org/10.1109/TIP.2002.804276)。
- Ronneberger et al. (2015), [U-Net](https://arxiv.org/abs/1505.04597)。
