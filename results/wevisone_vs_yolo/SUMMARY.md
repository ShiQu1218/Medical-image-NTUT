## 主要統計結論與適用範圍

主要外部評估使用 UCLM patient-held-out test（331 張／18 位病人）；UCLM all 含這些影像，兩者不可當作獨立重複驗證。

- **固定門檻 bootstrap，uclm:test**：WeVisOne gate − YOLO val-cal，病灶 Dice 差 +0.202（95% CI +0.116 至 +0.313）；正常影像 FPR 差 +5.7 個百分點（95% CI -0.3 至 +12.5），區間跨過 0，無法判定正常影像 FPR 有差異。這是各自驗證集校準後的固定操作點比較，並非相同 FPR 的比較。
- **固定門檻 bootstrap，uclm:all**：WeVisOne gate − YOLO val-cal，病灶 Dice 差 +0.227（95% CI +0.169 至 +0.295）；正常影像 FPR 差 +6.1 個百分點（95% CI +1.3 至 +11.1），WeVisOne 的正常影像 FPR 較高。這是各自驗證集校準後的固定操作點比較，並非相同 FPR 的比較。

bootstrap 先對每張影像跨訓練 seed 取平均，再以病人為單位做 5,000 次成對重抽樣；每次都保留原模型與門檻。CI 僅反映這些固定模型／門檻下的病人抽樣不確定性，未涵蓋重訓與重新校準，也未作多重比較校正。

- **測試集操作曲線，uclm:test，FPR ≤ 10%**：病灶 Dice 為 WeVisOne 0.698 ± 0.022、YOLO 0.504 ± 0.026（跨 seed mean ± SD）。各模型實際 FPR 為 9.6 ± 0.0%／9.4 ± 0.3%；這是相同 FPR 上限下的比較，實際 FPR 不必相等。
- **測試集操作曲線，uclm:all，FPR ≤ 10%**：病灶 Dice 為 WeVisOne 0.582 ± 0.006、YOLO 0.363 ± 0.004（跨 seed mean ± SD）。各模型實際 FPR 為 9.6 ± 0.3%／9.8 ± 0.0%；這是相同 FPR 上限下的比較，實際 FPR 不必相等。

操作曲線在測試集用正常標籤限制 FPR，再用測試病灶 Dice 選出最佳門檻；因此屬探索性、可能樂觀的包絡線。可說 WeVisOne 在這些測試集的相同 FPR 上限下呈現較高 Dice；不可引用固定門檻的 bootstrap CI 宣稱 matched-FPR 優勢達統計顯著，也不可將這些測試集門檻當作已驗證的部署規則。

定位補充見 [嚴格 F1](STRICT_F1.md)（union-mask IoU ≥ 0.25／0.50）及 [共同命中分析](COMMON_HITS.md)。共同命中的 Dice 受模型依賴的樣本選擇影響，需連同兩模型各自命中／漏失數與全病灶 Dice 解讀；這是影像級評估，並非病灶實例一對一匹配。

少量資料 WeVisOne：已通過完整性檢查 9/9 個 fraction × seed 組合；只有訓練與三個 split 評估皆完整的 run 會納入報告。目標為每個 10%／25%／50% 子集各有 seed 42、43、44；未完成前不可宣稱三 seed 結果已齊全。
