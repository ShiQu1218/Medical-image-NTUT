# WeVisOne-VL-4B vs YOLO26m-seg

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


FPR = normal images with any predicted pixel. miss = lesion images with zero GT overlap. AUROC = image-level normal vs lesion from the presence score (WeVisOne logit gap, YOLO max conf). Fixed-point tables use each model's own validation-calibrated threshold or a prespecified default (WeVisOne raw / YOLO conf >= 0.25). Matched-FPR curves and sensitivity-at-FPR use test-selected thresholds and are descriptive only. mean ± sample SD over training seeds; seed-pooled k/N repeats the same images.

Strict localization F1 and paired Dice on commonly hit lesions: [STRICT_F1.md](STRICT_F1.md), [COMMON_HITS.md](COMMON_HITS.md).

## 1. Trained on BUSI only
### In-domain and patient-held-out

| Model | busi:test FPR % | busi:test FPR excl. icon % | busi:test lesion Dice | busi:test miss % | busi:test AUROC | uclm:test FPR % | uclm:test FPR excl. icon % | uclm:test lesion Dice | uclm:test miss % | uclm:test AUROC | uclm:all FPR % | uclm:all FPR excl. icon % | uclm:all lesion Dice | uclm:all miss % | uclm:all AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WeVisOne raw (n=3) | 46.7 ± 5.8 (14/30) | 46.7 ± 5.8 | 0.829 ± 0.010 | 1.5 ± 0.0 (3/198) | 0.976 ± 0.012 | 17.7 ± 5.1 (100/564) | 14.9 ± 3.2 | 0.716 ± 0.022 | 9.8 ± 2.4 (42/429) | 0.954 ± 0.006 | 21.7 ± 5.5 (273/1257) | 18.6 ± 3.8 | 0.611 ± 0.017 | 16.7 ± 1.8 (130/780) | 0.937 ± 0.007 |
| WeVisOne + gate (n=3) | 40.0 ± 10.0 (12/30) | 40.0 ± 10.0 | 0.829 ± 0.010 | 1.5 ± 0.0 (3/198) | 0.976 ± 0.012 | 15.4 ± 6.5 (87/564) | 13.3 ± 4.6 | 0.708 ± 0.032 | 11.0 ± 3.6 (47/429) | 0.954 ± 0.006 | 18.9 ± 8.3 (237/1257) | 16.5 ± 6.4 | 0.607 ± 0.023 | 17.4 ± 2.6 (136/780) | 0.937 ± 0.007 |
| WeVisOne strong-aug raw (n=3) | 36.7 ± 11.5 (11/30) | 36.7 ± 11.5 | 0.808 ± 0.007 | 2.5 ± 0.9 (5/198) | 0.977 ± 0.009 | 10.6 ± 2.1 (60/564) | 10.3 ± 1.9 | 0.691 ± 0.002 | 10.7 ± 0.4 (46/429) | 0.955 ± 0.007 | 13.2 ± 2.5 (166/1257) | 12.9 ± 2.3 | 0.577 ± 0.008 | 18.3 ± 0.9 (143/780) | 0.935 ± 0.006 |
| WeVisOne strong-aug + gate (n=3) | 30.0 ± 10.0 (9/30) | 30.0 ± 10.0 | 0.805 ± 0.009 | 3.0 ± 1.5 (6/198) | 0.977 ± 0.009 | 9.4 ± 3.2 (53/564) | 9.0 ± 3.0 | 0.690 ± 0.004 | 11.9 ± 1.4 (51/429) | 0.955 ± 0.007 | 11.9 ± 3.5 (149/1257) | 11.6 ± 3.3 | 0.572 ± 0.006 | 20.1 ± 1.6 (157/780) | 0.935 ± 0.006 |
| YOLO conf≥0.25 (n=3) | 16.7 ± 28.9 (5/30) | 16.7 ± 28.9 | 0.774 ± 0.005 | 7.1 ± 2.3 (14/198) | 0.960 ± 0.024 | 40.2 ± 22.6 (227/564) | 22.7 ± 16.1 | 0.583 ± 0.068 | 29.8 ± 7.1 (128/429) | 0.820 ± 0.004 | 44.5 ± 21.8 (559/1257) | 27.5 ± 17.0 | 0.476 ± 0.070 | 40.5 ± 8.8 (316/780) | 0.731 ± 0.015 |
| YOLO val-calibrated (n=3) | 0.0 ± 0.0 (0/30) | 0.0 ± 0.0 | 0.704 ± 0.045 | 20.2 ± 5.3 (40/198) | 0.960 ± 0.024 | 9.8 ± 2.7 (55/564) | 6.9 ± 2.4 | 0.506 ± 0.011 | 43.4 ± 0.0 (186/429) | 0.820 ± 0.004 | 12.7 ± 3.1 (160/1257) | 9.3 ± 2.5 | 0.380 ± 0.005 | 56.3 ± 1.2 (439/780) | 0.731 ± 0.015 |
| YOLO conf≥0.25, icon dets dropped (n=3) | 16.7 ± 28.9 (5/30) | 16.7 ± 28.9 | 0.774 ± 0.005 | 7.1 ± 2.3 (14/198) | 0.960 ± 0.024 | 22.7 ± 16.1 (128/564) | 22.7 ± 16.1 | 0.594 ± 0.057 | 29.8 ± 7.1 (128/429) | 0.822 ± 0.005 | 27.5 ± 17.0 (346/1257) | 27.5 ± 17.0 | 0.486 ± 0.060 | 40.5 ± 8.8 (316/780) | 0.745 ± 0.013 |
| YOLO val-calibrated, icon dets dropped (n=3) | 0.0 ± 0.0 (0/30) | 0.0 ± 0.0 | 0.704 ± 0.045 | 20.2 ± 5.3 (40/198) | 0.960 ± 0.024 | 6.9 ± 2.4 (39/564) | 6.9 ± 2.4 | 0.508 ± 0.010 | 43.4 ± 0.0 (186/429) | 0.822 ± 0.005 | 9.3 ± 2.5 (117/1257) | 9.3 ± 2.5 | 0.381 ± 0.008 | 56.3 ± 1.2 (439/780) | 0.745 ± 0.013 |
| Hybrid: YOLO cal boxes → WeVisOne decoder (n=3) | 0.0 ± 0.0 (0/30) | 0.0 ± 0.0 | 0.707 ± 0.045 | 20.2 ± 5.3 (40/198) | 0.960 ± 0.024 | 9.8 ± 2.7 (55/564) | 6.9 ± 2.4 | 0.513 ± 0.010 | 43.4 ± 0.0 (186/429) | 0.820 ± 0.004 | 12.7 ± 3.1 (160/1257) | 9.3 ± 2.5 | 0.384 ± 0.005 | 56.4 ± 1.4 (440/780) | 0.731 ± 0.015 |
| Hybrid: YOLO 0.25 boxes → WeVisOne decoder (n=3) | 16.7 ± 28.9 (5/30) | 16.7 ± 28.9 | 0.778 ± 0.004 | 7.1 ± 2.3 (14/198) | 0.960 ± 0.024 | 40.2 ± 22.6 (227/564) | 22.5 ± 16.3 | 0.592 ± 0.069 | 29.8 ± 7.1 (128/429) | 0.820 ± 0.004 | 44.5 ± 21.8 (559/1257) | 27.4 ± 17.2 | 0.483 ± 0.072 | 40.5 ± 8.8 (316/780) | 0.731 ± 0.015 |

### Lesion-only external sets (no normals: FPR/AUROC undefined; BrEaST has 4 normals)

| Model | busuc:all FPR % | busuc:all FPR excl. icon % | busuc:all lesion Dice | busuc:all miss % | busuc:all AUROC | busbra:test FPR % | busbra:test FPR excl. icon % | busbra:test lesion Dice | busbra:test miss % | busbra:test AUROC | breast:all FPR % | breast:all FPR excl. icon % | breast:all lesion Dice | breast:all miss % | breast:all AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WeVisOne raw (n=3) | – | – | 0.880 ± 0.001 | 0.9 ± 0.1 (22/2433) | – | – | – | 0.733 ± 0.009 | 4.4 ± 0.7 (49/1122) | – | 41.7 ± 14.4 (5/12) | 41.7 ± 14.4 | 0.701 ± 0.011 | 13.2 ± 1.3 (100/756) | 0.902 ± 0.023 |
| WeVisOne + gate (n=3) | – | – | 0.879 ± 0.001 | 1.0 ± 0.3 (25/2433) | – | – | – | 0.732 ± 0.008 | 4.6 ± 0.8 (52/1122) | – | 33.3 ± 14.4 (4/12) | 33.3 ± 14.4 | 0.694 ± 0.015 | 14.4 ± 1.6 (109/756) | 0.902 ± 0.023 |
| WeVisOne strong-aug raw (n=3) | – | – | 0.823 ± 0.017 | 0.9 ± 0.1 (21/2433) | – | – | – | 0.642 ± 0.014 | 3.7 ± 0.4 (41/1122) | – | 41.7 ± 14.4 (5/12) | 41.7 ± 14.4 | 0.647 ± 0.025 | 13.5 ± 2.4 (102/756) | 0.878 ± 0.024 |
| WeVisOne strong-aug + gate (n=3) | – | – | 0.822 ± 0.017 | 0.9 ± 0.1 (23/2433) | – | – | – | 0.641 ± 0.015 | 4.4 ± 0.7 (49/1122) | – | 41.7 ± 14.4 (5/12) | 41.7 ± 14.4 | 0.638 ± 0.018 | 15.6 ± 2.4 (118/756) | 0.878 ± 0.024 |
| YOLO conf≥0.25 (n=3) | – | – | 0.625 ± 0.021 | 19.4 ± 9.4 (473/2433) | – | – | – | 0.398 ± 0.089 | 48.8 ± 17.4 (547/1122) | – | 33.3 ± 38.2 (4/12) | 33.3 ± 38.2 | 0.443 ± 0.076 | 45.2 ± 11.7 (342/756) | 0.614 ± 0.098 |
| YOLO val-calibrated (n=3) | – | – | 0.502 ± 0.052 | 42.5 ± 6.4 (1034/2433) | – | – | – | 0.291 ± 0.034 | 66.2 ± 4.7 (743/1122) | – | 16.7 ± 14.4 (2/12) | 16.7 ± 14.4 | 0.335 ± 0.012 | 61.5 ± 1.6 (465/756) | 0.614 ± 0.098 |
| YOLO conf≥0.25, icon dets dropped (n=3) | – | – | 0.625 ± 0.021 | 19.4 ± 9.4 (473/2433) | – | – | – | 0.398 ± 0.089 | 48.8 ± 17.4 (547/1122) | – | 33.3 ± 38.2 (4/12) | 33.3 ± 38.2 | 0.443 ± 0.076 | 45.2 ± 11.7 (342/756) | 0.614 ± 0.098 |
| YOLO val-calibrated, icon dets dropped (n=3) | – | – | 0.502 ± 0.052 | 42.5 ± 6.4 (1034/2433) | – | – | – | 0.291 ± 0.034 | 66.2 ± 4.7 (743/1122) | – | 16.7 ± 14.4 (2/12) | 16.7 ± 14.4 | 0.335 ± 0.012 | 61.5 ± 1.6 (465/756) | 0.614 ± 0.098 |
| Hybrid: YOLO cal boxes → WeVisOne decoder (n=3) | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| Hybrid: YOLO 0.25 boxes → WeVisOne decoder (n=3) | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |

### Lesion performance at matched normal-image FPR (thresholds swept on the test set; analysis, not a deployable rule)

For each test cohort and seed, maximize TEST lesion Dice among all evaluated thresholds with TEST FPR <= the budget. This uses both test normal labels and lesion masks, so values are selected descriptive envelopes and can be optimistic. Actual FPR can be below the budget and differs across models; exact achieved FPRs and thresholds are in report.json. These comparisons do not inherit the fixed-point bootstrap CIs or establish statistical superiority at matched FPR.

| Set | Model | runs | FPR ≤ 5%: lesion Dice / miss % | FPR ≤ 10% | FPR ≤ 15% |
|---|---|---:|---|---|---|
| uclm:all | WeVisOne (presence gate) | 3 | 0.527 ± 0.014 / 29.9 ± 2.8 | 0.582 ± 0.006 / 20.9 ± 2.1 | 0.603 ± 0.011 / 17.9 ± 1.2 |
| uclm:all | YOLO (conf threshold) | 3 | 0.324 ± 0.002 / 64.2 ± 0.7 | 0.363 ± 0.004 / 58.8 ± 1.2 | 0.405 ± 0.017 / 53.1 ± 1.7 |
| uclm:all | nnU-Net (max-prob gate) | 1 | 0.160 / 81.2 | 0.253 / 68.8 | 0.315 / 60.0 |
| uclm:test | WeVisOne (presence gate) | 3 | 0.658 ± 0.010 / 18.4 ± 1.6 | 0.698 ± 0.022 / 12.8 ± 2.5 | 0.710 ± 0.019 / 10.7 ± 2.1 |
| uclm:test | YOLO (conf threshold) | 3 | 0.476 ± 0.023 / 47.6 ± 3.5 | 0.504 ± 0.026 / 44.1 ± 3.5 | 0.540 ± 0.036 / 38.9 ± 4.0 |
| uclm:test | nnU-Net (max-prob gate) | 1 | 0.314 / 63.6 | 0.386 / 55.2 | 0.429 / 48.3 |

## 2. GT-box oracle (seed 42): where is the external gap?

| Set | generate lesion Dice | GT-box lesion Dice | generate miss % |
|---|---:|---:|---:|
| busi:test | 0.833 | 0.919 | 1.5 |
| uclm:all | 0.592 | 0.908 | 17.7 |
| uclm:test | 0.693 | 0.924 | 11.2 |

## 3. Grouped 5-fold CV, out-of-fold over all 529 clean BUSI images (72 normals)

| Model | op | FPR % | lesion Dice | miss % | AUROC | sens@FPR10% |
|---|---|---:|---:|---:|---:|---:|
| wevis | raw | 26.4 (19/72) | 0.807 | 5.3 | 0.975 | 0.934 |
| wevis | gate | 13.9 (10/72) | 0.760 | 11.2 | 0.975 | 0.934 |
| yolo | 0.25 | 18.1 (13/72) | 0.738 | 10.9 | 0.944 | 0.869 |
| yolo | cal | 6.9 (5/72) | 0.699 | 18.2 | 0.944 | 0.869 |
| yolo | 0.25_noicon | 18.1 (13/72) | 0.738 | 10.9 | 0.944 | 0.869 |
| yolo | cal_noicon | 6.9 (5/72) | 0.699 | 18.2 | 0.944 | 0.869 |

## 4. Low-data regime (fraction of the 368 BUSI training images)
Training/subset seeds = 42/43/44, paired with YOLO; nested subsets per seed, 230 WeVisOne updates and 5 validations. Only fully audited WeVisOne runs enter this table. Completed low-data runs: 9/9; see low_data_status.json.

### 

| Model | busi:test FPR % | busi:test FPR excl. icon % | busi:test lesion Dice | busi:test miss % | busi:test AUROC | uclm:all FPR % | uclm:all FPR excl. icon % | uclm:all lesion Dice | uclm:all miss % | uclm:all AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% WeVisOne gate (n=3) | 13.3 ± 15.3 (4/30) | 13.3 ± 15.3 | 0.416 ± 0.102 | 51.5 ± 10.5 (102/198) | 0.817 ± 0.148 | 3.0 ± 1.6 (38/1257) | 3.0 ± 1.6 | 0.301 ± 0.079 | 60.6 ± 7.6 (473/780) | 0.872 ± 0.028 |
| 10% YOLO cal (n=3) | 10.0 ± 10.0 (3/30) | 10.0 ± 10.0 | 0.069 ± 0.035 | 88.4 ± 8.6 (175/198) | 0.564 ± 0.081 | 7.3 ± 5.9 (92/1257) | 6.8 ± 5.6 | 0.032 ± 0.035 | 92.2 ± 6.5 (719/780) | 0.601 ± 0.033 |
| 25% WeVisOne gate (n=3) | 23.3 ± 20.8 (7/30) | 23.3 ± 20.8 | 0.740 ± 0.107 | 12.1 ± 14.5 (24/198) | 0.948 ± 0.029 | 25.3 ± 19.5 (318/1257) | 18.4 ± 13.6 | 0.534 ± 0.112 | 26.5 ± 19.1 (207/780) | 0.935 ± 0.014 |
| 25% YOLO cal (n=3) | 10.0 ± 17.3 (3/30) | 10.0 ± 17.3 | 0.650 ± 0.051 | 22.2 ± 7.2 (44/198) | 0.941 ± 0.010 | 26.3 ± 17.1 (331/1257) | 19.2 ± 4.9 | 0.391 ± 0.021 | 51.8 ± 2.5 (404/780) | 0.729 ± 0.008 |
| 50% WeVisOne gate (n=3) | 33.3 ± 5.8 (10/30) | 33.3 ± 5.8 | 0.807 ± 0.021 | 3.0 ± 3.0 (6/198) | 0.975 ± 0.006 | 20.8 ± 9.1 (261/1257) | 15.6 ± 8.4 | 0.600 ± 0.056 | 19.1 ± 7.1 (149/780) | 0.929 ± 0.008 |
| 50% YOLO cal (n=3) | 6.7 ± 11.5 (2/30) | 6.7 ± 11.5 | 0.706 ± 0.013 | 17.2 ± 4.9 (34/198) | 0.957 ± 0.014 | 20.7 ± 21.6 (260/1257) | 16.2 ± 14.2 | 0.434 ± 0.063 | 47.6 ± 10.7 (371/780) | 0.773 ± 0.030 |
| 100% WeVisOne gate (n=3) | 40.0 ± 10.0 (12/30) | 40.0 ± 10.0 | 0.829 ± 0.010 | 1.5 ± 0.0 (3/198) | 0.976 ± 0.012 | 18.9 ± 8.3 (237/1257) | 16.5 ± 6.4 | 0.607 ± 0.023 | 17.4 ± 2.6 (136/780) | 0.937 ± 0.007 |
| 100% YOLO cal (n=3) | 0.0 ± 0.0 (0/30) | 0.0 ± 0.0 | 0.704 ± 0.045 | 20.2 ± 5.3 (40/198) | 0.960 ± 0.024 | 12.7 ± 3.1 (160/1257) | 9.3 ± 2.5 | 0.380 ± 0.005 | 56.3 ± 1.2 (439/780) | 0.731 ± 0.015 |

## 5. Multi-source training (BUS_UC never used; UCLM test patients never used)
### 

| Model | busi:test FPR % | busi:test FPR excl. icon % | busi:test lesion Dice | busi:test miss % | busi:test AUROC | uclm:test FPR % | uclm:test FPR excl. icon % | uclm:test lesion Dice | uclm:test miss % | uclm:test AUROC | busuc:all FPR % | busuc:all FPR excl. icon % | busuc:all lesion Dice | busuc:all miss % | busuc:all AUROC | busbra:test FPR % | busbra:test FPR excl. icon % | busbra:test lesion Dice | busbra:test miss % | busbra:test AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BUSI only WeVisOne strong-aug + gate (n=3) | 30.0 ± 10.0 (9/30) | 30.0 ± 10.0 | 0.805 ± 0.009 | 3.0 ± 1.5 (6/198) | 0.977 ± 0.009 | 9.4 ± 3.2 (53/564) | 9.0 ± 3.0 | 0.690 ± 0.004 | 11.9 ± 1.4 (51/429) | 0.955 ± 0.007 | – | – | 0.822 ± 0.017 | 0.9 ± 0.1 (23/2433) | – | – | – | 0.641 ± 0.015 | 4.4 ± 0.7 (49/1122) | – |
| BUSI only YOLO cal (n=3) | 0.0 ± 0.0 (0/30) | 0.0 ± 0.0 | 0.704 ± 0.045 | 20.2 ± 5.3 (40/198) | 0.960 ± 0.024 | 9.8 ± 2.7 (55/564) | 6.9 ± 2.4 | 0.506 ± 0.011 | 43.4 ± 0.0 (186/429) | 0.820 ± 0.004 | – | – | 0.502 ± 0.052 | 42.5 ± 6.4 (1034/2433) | – | – | – | 0.291 ± 0.034 | 66.2 ± 4.7 (743/1122) | – |
| BUSI+BRA WeVisOne + gate (n=3) | 10.0 ± 0.0 (3/30) | 10.0 ± 0.0 | 0.795 ± 0.003 | 3.5 ± 1.7 (7/198) | 0.986 ± 0.005 | 7.1 ± 1.3 (40/564) | 7.1 ± 1.3 | 0.712 ± 0.026 | 16.6 ± 3.3 (71/429) | 0.969 ± 0.003 | – | – | 0.876 ± 0.007 | 0.6 ± 0.1 (14/2433) | – | – | – | 0.867 ± 0.004 | 0.3 ± 0.0 (3/1122) | – |
| BUSI+BRA YOLO cal (n=3) | 3.3 ± 5.8 (1/30) | 3.3 ± 5.8 | 0.698 ± 0.052 | 22.2 ± 7.0 (44/198) | 0.958 ± 0.024 | 5.1 ± 1.9 (29/564) | 5.1 ± 1.9 | 0.654 ± 0.030 | 26.3 ± 3.9 (113/429) | 0.927 ± 0.019 | – | – | 0.750 ± 0.041 | 17.7 ± 4.3 (430/2433) | – | – | – | 0.863 ± 0.000 | 4.8 ± 0.5 (54/1122) | – |
| BUSI+BRA+UCLMtrain WeVisOne + gate (n=3) | 3.3 ± 5.8 (1/30) | 3.3 ± 5.8 | 0.736 ± 0.067 | 14.6 ± 9.7 (29/198) | 0.977 ± 0.005 | 1.1 ± 1.4 (6/564) | 1.1 ± 1.4 | 0.312 ± 0.276 | 65.3 ± 30.8 (280/429) | 0.967 ± 0.005 | – | – | 0.872 ± 0.010 | 0.4 ± 0.2 (10/2433) | – | – | – | 0.864 ± 0.005 | 0.2 ± 0.2 (2/1122) | – |
| BUSI+BRA+UCLMtrain YOLO cal (n=3) | 6.7 ± 5.8 (2/30) | 6.7 ± 5.8 | 0.711 ± 0.048 | 21.2 ± 6.9 (42/198) | 0.937 ± 0.018 | 2.3 ± 0.8 (13/564) | 2.3 ± 0.8 | 0.688 ± 0.030 | 23.5 ± 3.6 (101/429) | 0.954 ± 0.007 | – | – | 0.790 ± 0.018 | 13.1 ± 2.1 (319/2433) | – | – | – | 0.869 ± 0.007 | 4.2 ± 0.9 (47/1122) | – |

## 5b. Authoritative baselines on the same split and metric
### nnU-Net v2 2d (nnUNetTrainer_250epochs, fold 0 = our train/val) vs WeVisOne vs YOLO

| Model | busi:test FPR % | busi:test FPR excl. icon % | busi:test lesion Dice | busi:test miss % | busi:test AUROC | uclm:test FPR % | uclm:test FPR excl. icon % | uclm:test lesion Dice | uclm:test miss % | uclm:test AUROC | uclm:all FPR % | uclm:all FPR excl. icon % | uclm:all lesion Dice | uclm:all miss % | uclm:all AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WeVisOne raw (n=3) | 46.7 ± 5.8 (14/30) | 46.7 ± 5.8 | 0.829 ± 0.010 | 1.5 ± 0.0 (3/198) | 0.976 ± 0.012 | 17.7 ± 5.1 (100/564) | 14.9 ± 3.2 | 0.716 ± 0.022 | 9.8 ± 2.4 (42/429) | 0.954 ± 0.006 | 21.7 ± 5.5 (273/1257) | 18.6 ± 3.8 | 0.611 ± 0.017 | 16.7 ± 1.8 (130/780) | 0.937 ± 0.007 |
| WeVisOne + gate (n=3) | 40.0 ± 10.0 (12/30) | 40.0 ± 10.0 | 0.829 ± 0.010 | 1.5 ± 0.0 (3/198) | 0.976 ± 0.012 | 15.4 ± 6.5 (87/564) | 13.3 ± 4.6 | 0.708 ± 0.032 | 11.0 ± 3.6 (47/429) | 0.954 ± 0.006 | 18.9 ± 8.3 (237/1257) | 16.5 ± 6.4 | 0.607 ± 0.023 | 17.4 ± 2.6 (136/780) | 0.937 ± 0.007 |
| nnU-Net raw (n=1) | 30.0 (3/10) | 30.0 | 0.823 | 0.0 (0/66) | 0.992 | 69.1 (130/188) | 64.9 | 0.584 | 21.7 (31/143) | 0.752 | 72.3 (303/419) | 69.5 | 0.493 | 26.2 (68/260) | 0.681 |
| nnU-Net + gate (val max-prob threshold) (n=1) | 0.0 (0/10) | 0.0 | 0.642 | 28.8 (19/66) | 0.992 | 3.2 (6/188) | 3.2 | 0.249 | 71.3 (102/143) | 0.752 | 4.1 (17/419) | 4.1 | 0.156 | 81.5 (212/260) | 0.681 |
| YOLO val-calibrated (n=3) | 0.0 ± 0.0 (0/30) | 0.0 ± 0.0 | 0.704 ± 0.045 | 20.2 ± 5.3 (40/198) | 0.960 ± 0.024 | 9.8 ± 2.7 (55/564) | 6.9 ± 2.4 | 0.506 ± 0.011 | 43.4 ± 0.0 (186/429) | 0.820 ± 0.004 | 12.7 ± 3.1 (160/1257) | 9.3 ± 2.5 | 0.380 ± 0.005 | 56.3 ± 1.2 (439/780) | 0.731 ± 0.015 |

### 

| Model | busuc:all FPR % | busuc:all FPR excl. icon % | busuc:all lesion Dice | busuc:all miss % | busuc:all AUROC | busbra:test FPR % | busbra:test FPR excl. icon % | busbra:test lesion Dice | busbra:test miss % | busbra:test AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WeVisOne raw (n=3) | – | – | 0.880 ± 0.001 | 0.9 ± 0.1 (22/2433) | – | – | – | 0.733 ± 0.009 | 4.4 ± 0.7 (49/1122) | – |
| nnU-Net raw (n=1) | – | – | 0.819 | 0.9 (7/811) | – | – | – | 0.677 | 11.8 (44/374) | – |
| YOLO conf≥0.25 (n=3) | – | – | 0.625 ± 0.021 | 19.4 ± 9.4 (473/2433) | – | – | – | 0.398 ± 0.089 | 48.8 ± 17.4 (547/1122) | – |

### Box-prompted decoders (lesion Dice; MedSAM zero-shot, likely saw BUSI in pre-training)

| Set | MedSAM, GT boxes | WeVisOne decoder, GT boxes | MedSAM, YOLO-cal boxes | WeVisOne decoder, YOLO-cal boxes |
|---|---:|---:|---:|---:|
| busi:test | 0.872 | 0.919 | 0.681 | 0.707 |
| uclm:all | 0.784 | 0.908 | 0.355 | 0.384 |
| uclm:test | 0.805 | 0.924 | 0.473 | 0.513 |
| busuc:all | 0.899 | – | – | – |
| busbra:test | 0.845 | – | – | – |

## 5c. Replicating Yu et al. 2026 (QIMS) YOLO protocol
Uncleaned BUSI lesions (647), image-level random 80/10/10, yolo11n-seg, 512 px, 50 epochs, conf ≥ 0.25. 'leaked' = test image whose author-curated duplicate group also appears in train.

| split seed | test per-image Dice | test pooled Dice | leaked (n) per-image | not leaked (n) per-image | UCLM lesions per-image | UCLM lesions pooled | UCLM normal FPR % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.741 | 0.703 | 0.814 (18) | 0.712 (46) | 0.562 | 0.679 | 38.7 |
| 1 | 0.782 | 0.762 | 0.790 (31) | 0.775 (33) | 0.562 | 0.624 | 72.6 |
| 2 | 0.804 | 0.809 | 0.870 (22) | 0.769 (42) | 0.580 | 0.696 | 32.7 |

## 5d. Synthetic domain-shift stress test (BUSI test 76 / UCLM held-out patients 331; seed 42 models)
Families gain/speckle/blur overlap the strong-aug training augmentation (marked *). Thresholds fixed on BUSI val. Cells: BUSI lesion Dice / UCLM lesion Dice / UCLM normal FPR %.

| Corruption | WeVisOne gate | WeVisOne strong-aug gate | YOLO cal | nnU-Net gate |
|---|---|---|---|---|
| clean | 0.833 / 0.674 / 8.0 | 0.815 / 0.692 / 9.0 | 0.748 / 0.498 / 12.2 | 0.642 / 0.249 / 3.2 |
| gain1* | 0.804 / 0.672 / 7.4 | 0.797 / 0.686 / 9.0 | 0.698 / 0.475 / 5.3 | 0.565 / 0.219 / 3.2 |
| gain2* | 0.818 / 0.633 / 5.9 | 0.785 / 0.629 / 6.4 | 0.670 / 0.391 / 3.2 | 0.596 / 0.197 / 2.7 |
| speckle1* | 0.797 / 0.667 / 11.2 | 0.754 / 0.668 / 11.2 | 0.170 / 0.159 / 93.6 | 0.545 / 0.248 / 3.2 |
| speckle2* | 0.779 / 0.693 / 11.7 | 0.748 / 0.668 / 12.2 | 0.174 / 0.164 / 98.4 | 0.491 / 0.294 / 3.2 |
| blur1* | 0.812 / 0.646 / 3.7 | 0.795 / 0.680 / 10.6 | 0.692 / 0.518 / 6.4 | 0.599 / 0.261 / 5.3 |
| blur2* | 0.785 / 0.697 / 8.5 | 0.770 / 0.727 / 13.3 | 0.619 / 0.490 / 10.6 | 0.645 / 0.267 / 6.4 |
| tgc1 | 0.832 / 0.659 / 6.9 | 0.821 / 0.660 / 8.0 | 0.751 / 0.473 / 10.1 | 0.619 / 0.246 / 3.7 |
| tgc2 | 0.841 / 0.636 / 7.4 | 0.803 / 0.659 / 8.5 | 0.746 / 0.487 / 9.6 | 0.556 / 0.203 / 3.2 |
| jpeg1 | 0.823 / 0.690 / 7.4 | 0.799 / 0.710 / 11.7 | 0.755 / 0.508 / 32.4 | 0.642 / 0.243 / 3.2 |
| jpeg2 | 0.802 / 0.714 / 21.8 | 0.755 / 0.713 / 15.4 | 0.714 / 0.454 / 44.1 | 0.614 / 0.239 / 3.7 |
| icon1 | 0.826 / 0.707 / 8.5 | 0.779 / 0.686 / 11.7 | 0.717 / 0.465 / 36.7 | 0.615 / 0.245 / 3.2 |
| icon2 | 0.821 / 0.695 / 29.8 | 0.775 / 0.665 / 26.6 | 0.636 / 0.447 / 42.6 | 0.548 / 0.183 / 11.7 |
| sector1 | 0.821 / 0.692 / 14.9 | 0.805 / 0.684 / 9.6 | 0.485 / 0.415 / 44.1 | 0.501 / 0.229 / 5.3 |
| sector2 | 0.794 / 0.643 / 23.0 | 0.795 / 0.649 / 8.9 | 0.578 / 0.283 / 86.4 | 0.351 / 0.237 / 2.6 |
| fov1 | 0.826 / 0.716 / 20.2 | 0.830 / 0.702 / 10.6 | 0.795 / 0.527 / 3.2 | 0.577 / 0.259 / 1.6 |
| fov2 | 0.807 / 0.656 / 94.1 | 0.780 / 0.665 / 21.3 | 0.711 / 0.481 / 1.6 | 0.619 / 0.313 / 2.1 |

## 6. Paired patient-cluster bootstrap (A − B; 5000 resamples)

FIXED operating-point analysis: predictions are generated with validation-calibrated or prespecified default thresholds. Average each image's metric across the available training seeds, then resample patients with replacement, using the same sampled patients for A and B. Thresholds and models stay fixed in every resample. Percentile 95% CIs describe patient-sampling uncertainty conditional on these models and thresholds; they exclude retraining and calibration uncertainty, are not adjusted for multiple comparisons, and do not apply to matched-FPR curves. Positive ΔFPR is worse for A; positive ΔDice is better for A. UCLM test is a subset of UCLM all, not an independent replication. UCLM has patient IDs; BUSI and BUS_UC use available image/duplicate groups as resampling units, not verified patients.

| Comparison | images / resampling units | ΔFPR (pp) [95% CI] | Δ lesion Dice [95% CI] |
|---|---:|---:|---:|
| uclm:all / WeVisOne gate vs YOLO cal | 679 / 38 | +6.1 [+1.3, +11.1] | +0.227 [+0.169, +0.295] |
| uclm:all / WeVisOne raw vs YOLO 0.25 | 679 / 38 | -22.8 [-27.9, -17.3] | +0.135 [+0.080, +0.198] |
| uclm:all / WeVisOne aug gate vs YOLO cal | 679 / 38 | -0.9 [-5.0, +3.5] | +0.192 [+0.133, +0.259] |
| uclm:test / WeVisOne gate vs YOLO cal | 331 / 18 | +5.7 [-0.3, +12.5] | +0.202 [+0.116, +0.313] |
| uclm:test / WeVisOne raw vs YOLO 0.25 | 331 / 18 | -22.5 [-29.3, -15.6] | +0.133 [+0.050, +0.238] |
| uclm:test / WeVisOne aug gate vs YOLO cal | 331 / 18 | -0.4 [-5.4, +6.1] | +0.184 [+0.096, +0.300] |
| busuc:all / WeVisOne gate vs YOLO cal | 811 / 811 | – | +0.378 [+0.352, +0.403] |
| busuc:all / WeVisOne raw vs YOLO 0.25 | 811 / 811 | – | +0.254 [+0.234, +0.275] |
| busuc:all / WeVisOne aug gate vs YOLO cal | 811 / 811 | – | +0.321 [+0.296, +0.345] |
| BUSI OOF / WeVisOne gate vs YOLO cal | 529 / 529 | +6.9 [-2.7, +16.9] | +0.061 [+0.030, +0.090] |
| BUSI OOF / WeVisOne raw vs YOLO 0.25 | 529 / 529 | +8.3 [-5.4, +21.7] | +0.069 [+0.044, +0.095] |
