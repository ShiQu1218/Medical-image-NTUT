# Stratified lesion analysis and HD95 (BUSI-trained models, fixed val operating points)

Models (BUSI-trained, fixed validation operating points, same as REPORT.md section 1/5b):
  WeVisOne gate (baseline_s42/43/44), YOLO val-cal (busi_m_s42/43/44), nnU-Net gate (busi_s0..s2 as available).
Lesion images only. Strata: pathology (benign/malignant) and GT area fraction at 256x256
(small < 3%, medium 3-10%, large >= 10%; fixed, label-only bins).
Per stratum: lesion Dice (misses count as 0), miss % (zero GT overlap),
HD95 in 256-grid pixels over images with a nonempty prediction (median, since HD95 is heavy-tailed).
HD95 is conditional on the model producing a mask, so a model that misses more hard lesions is scored
on an easier subset; read it together with miss %.
HD95 is in resampled pixels, not mm: cohorts have different physical spacing, so compare models
within a cohort, not cohorts with each other. mean ± SD across seeds (nnU-Net n=1).
Writes runs/comparison/STRATIFIED.md + stratified.json.

Cells: lesion Dice / miss % / median HD95 (px @256).

## busi:test

| Model | all (n=66) | benign (n=44) | malignant (n=22) | small <3% (n=22) | medium 3-10% (n=21) | large ≥10% (n=23) |
|---|---|---|---|---|---|---|
| WeVisOne gate | 0.829 ± 0.010 / 1.5 ± 0.0 / 5.3 ± 0.3 | 0.866 ± 0.006 / 2.3 ± 0.0 / 3.9 ± 0.1 | 0.757 ± 0.023 / 0.0 ± 0.0 / 21.4 ± 2.5 | 0.874 ± 0.009 / 0.0 ± 0.0 / 2.4 ± 0.2 | 0.814 ± 0.007 / 4.8 ± 0.0 / 5.5 ± 0.4 | 0.801 ± 0.019 / 0.0 ± 0.0 / 21.1 ± 1.2 |
| YOLO val-cal | 0.704 ± 0.045 / 20.2 ± 5.3 / 5.0 ± 0.8 | 0.787 ± 0.023 / 12.9 ± 2.6 / 3.9 ± 0.2 | 0.539 ± 0.096 / 34.8 ± 11.4 / 11.5 ± 1.3 | 0.787 ± 0.025 / 10.6 ± 2.6 / 3.1 ± 0.2 | 0.702 ± 0.036 / 23.8 ± 4.8 / 4.2 ± 0.5 | 0.628 ± 0.101 / 26.1 ± 11.5 / 14.8 ± 2.5 |
| nnU-Net gate | 0.388 ± 0.286 / 56.6 ± 31.6 / 3.7 ± 0.2 | 0.456 ± 0.323 / 49.2 ± 35.3 / 3.2 ± 0.0 | 0.254 ± 0.219 / 71.2 ± 25.0 / 9.9 ± 5.1 | 0.615 ± 0.330 / 31.8 ± 35.5 / 3.1 ± 0.4 | 0.344 ± 0.330 / 61.9 ± 36.0 / 4.6 ± 0.6 | 0.212 ± 0.227 / 75.4 ± 26.2 / 21.6 ± 1.4 |

Common hits (both WeVisOne gate and YOLO val-cal overlap GT, same seed; n = 53 ± 4 lesions): median HD95 WeVisOne 4.7 ± 0.4 vs YOLO 5.0 ± 0.7 px; Dice 0.877 vs 0.883.

## uclm:test

| Model | all (n=143) | benign (n=115) | malignant (n=28) | small <3% (n=46) | medium 3-10% (n=65) | large ≥10% (n=32) |
|---|---|---|---|---|---|---|
| WeVisOne gate | 0.708 ± 0.032 / 11.0 ± 3.6 / 6.4 ± 0.4 | 0.745 ± 0.026 / 8.7 ± 3.1 / 5.9 ± 0.0 | 0.557 ± 0.059 / 20.2 ± 5.5 / 9.7 ± 2.1 | 0.639 ± 0.057 / 15.2 ± 5.8 / 5.3 ± 0.2 | 0.741 ± 0.018 / 12.8 ± 3.2 / 5.4 ± 0.4 | 0.741 ± 0.028 / 1.0 ± 1.8 / 24.5 ± 9.9 |
| YOLO val-cal | 0.506 ± 0.011 / 43.4 ± 0.0 / 6.5 ± 0.8 | 0.587 ± 0.005 / 34.8 ± 0.9 / 6.2 ± 0.9 | 0.172 ± 0.046 / 78.6 ± 3.6 / 43.9 ± 55.4 | 0.396 ± 0.042 / 54.3 ± 5.8 / 4.0 ± 0.0 | 0.578 ± 0.051 / 35.9 ± 4.4 / 6.8 ± 1.8 | 0.519 ± 0.012 / 42.7 ± 1.8 / 14.1 ± 6.2 |
| nnU-Net gate | 0.130 ± 0.122 / 85.1 ± 14.0 / 4.2 ± 1.1 | 0.158 ± 0.147 / 81.7 ± 17.0 / 4.1 ± 1.1 | 0.011 ± 0.019 / 98.8 ± 2.1 / 130.4 | 0.186 ± 0.149 / 77.5 ± 18.1 / 3.4 ± 0.7 | 0.125 ± 0.142 / 86.2 ± 15.6 / 6.6 ± 1.5 | 0.058 ± 0.058 / 93.8 ± 6.2 / 47.3 ± 58.2 |

Common hits (both WeVisOne gate and YOLO val-cal overlap GT, same seed; n = 77 ± 2 lesions): median HD95 WeVisOne 4.8 ± 0.3 vs YOLO 6.3 ± 0.6 px; Dice 0.881 vs 0.896.

## uclm:all

| Model | all (n=260) | benign (n=170) | malignant (n=90) | small <3% (n=93) | medium 3-10% (n=102) | large ≥10% (n=65) |
|---|---|---|---|---|---|---|
| WeVisOne gate | 0.607 ± 0.023 / 17.4 ± 2.6 / 10.6 ± 0.4 | 0.676 ± 0.028 / 13.1 ± 3.0 / 6.9 ± 0.2 | 0.477 ± 0.017 / 25.6 ± 1.9 / 26.5 ± 1.4 | 0.587 ± 0.038 / 18.3 ± 3.9 / 6.3 ± 0.1 | 0.618 ± 0.018 / 23.5 ± 3.5 / 7.0 ± 0.4 | 0.617 ± 0.012 / 6.7 ± 0.9 / 31.7 ± 9.0 |
| YOLO val-cal | 0.380 ± 0.005 / 56.3 ± 1.2 / 8.1 ± 1.9 | 0.486 ± 0.022 / 44.7 ± 3.1 / 6.3 ± 0.7 | 0.178 ± 0.042 / 78.1 ± 3.4 / 22.7 ± 12.2 | 0.290 ± 0.051 / 65.6 ± 6.0 / 3.8 ± 0.7 | 0.443 ± 0.045 / 50.7 ± 4.1 / 7.1 ± 1.7 | 0.409 ± 0.012 / 51.8 ± 2.4 / 22.1 ± 8.2 |
| nnU-Net gate | 0.084 ± 0.075 / 89.9 ± 8.9 / 4.4 ± 1.5 | 0.126 ± 0.111 / 84.9 ± 13.1 / 4.4 ± 1.5 | 0.005 ± 0.009 / 99.3 ± 1.3 / 130.9 | 0.126 ± 0.095 / 83.9 ± 12.2 / 3.7 ± 0.9 | 0.081 ± 0.093 / 90.8 ± 10.5 / 6.7 ± 1.3 | 0.028 ± 0.028 / 96.9 ± 3.1 / 47.3 ± 58.2 |

Common hits (both WeVisOne gate and YOLO val-cal overlap GT, same seed; n = 108 ± 5 lesions): median HD95 WeVisOne 5.5 ± 0.7 vs YOLO 7.1 ± 1.1 px; Dice 0.829 vs 0.872.

## busuc:all

| Model | all (n=811) | benign (n=358) | malignant (n=453) | small <3% (n=54) | medium 3-10% (n=302) | large ≥10% (n=455) |
|---|---|---|---|---|---|---|
| WeVisOne gate | 0.879 ± 0.001 / 1.0 ± 0.3 / 8.0 ± 0.0 | 0.901 ± 0.002 / 0.4 ± 0.2 / 7.6 ± 0.3 | 0.862 ± 0.002 / 1.5 ± 0.4 / 8.6 ± 0.0 | 0.772 ± 0.003 / 3.7 ± 0.0 / 5.8 ± 0.5 | 0.858 ± 0.002 / 1.4 ± 0.5 / 7.2 ± 0.1 | 0.906 ± 0.003 / 0.4 ± 0.2 / 9.0 ± 0.0 |
| YOLO val-cal | 0.502 ± 0.052 / 42.5 ± 6.4 / 13.8 ± 0.4 | 0.537 ± 0.056 / 38.5 ± 6.9 / 14.0 ± 0.4 | 0.473 ± 0.050 / 45.7 ± 6.0 / 13.7 ± 1.1 | 0.141 ± 0.017 / 82.1 ± 2.8 / 7.5 ± 0.8 | 0.453 ± 0.037 / 47.6 ± 4.6 / 8.7 ± 0.6 | 0.577 ± 0.066 / 34.4 ± 8.0 / 17.1 ± 1.2 |
| nnU-Net gate | 0.352 ± 0.294 / 59.4 ± 33.9 / 10.1 ± 3.4 | 0.385 ± 0.331 / 56.0 ± 37.9 / 10.4 ± 2.3 | 0.326 ± 0.264 / 62.0 ± 30.8 / 9.7 ± 4.1 | 0.316 ± 0.107 / 61.1 ± 14.5 / 6.0 ± 1.6 | 0.390 ± 0.325 / 54.6 ± 37.8 / 8.3 ± 0.8 | 0.332 ± 0.296 / 62.3 ± 33.8 / 14.3 ± 1.8 |

Common hits (both WeVisOne gate and YOLO val-cal overlap GT, same seed; n = 466 ± 51 lesions): median HD95 WeVisOne 7.0 ± 0.2 vs YOLO 13.7 ± 0.4 px; Dice 0.920 vs 0.873.

## busbra:test

| Model | all (n=374) | benign (n=253) | malignant (n=121) | small <3% (n=85) | medium 3-10% (n=161) | large ≥10% (n=128) |
|---|---|---|---|---|---|---|
| WeVisOne gate | 0.732 ± 0.008 / 4.6 ± 0.8 / 13.0 ± 0.3 | 0.735 ± 0.006 / 4.2 ± 1.0 / 9.8 ± 0.8 | 0.727 ± 0.013 / 5.5 ± 1.0 / 19.0 ± 0.7 | 0.667 ± 0.009 / 4.3 ± 1.8 / 10.2 ± 0.8 | 0.729 ± 0.010 / 5.6 ± 0.6 / 11.1 ± 0.3 | 0.779 ± 0.010 / 3.6 ± 1.2 / 15.9 ± 0.4 |
| YOLO val-cal | 0.291 ± 0.034 / 66.2 ± 4.7 / 10.2 ± 0.4 | 0.341 ± 0.045 / 60.5 ± 5.9 / 8.5 ± 1.2 | 0.185 ± 0.015 / 78.2 ± 2.5 / 18.5 ± 2.4 | 0.225 ± 0.060 / 71.8 ± 8.2 / 6.0 ± 1.4 | 0.277 ± 0.064 / 68.3 ± 7.6 / 7.9 ± 1.0 | 0.351 ± 0.021 / 59.9 ± 2.3 / 15.5 ± 3.2 |
| nnU-Net gate | 0.172 ± 0.140 / 79.5 ± 16.4 / 8.0 ± 1.1 | 0.208 ± 0.166 / 75.4 ± 19.1 / 6.8 ± 0.2 | 0.095 ± 0.085 / 88.2 ± 10.7 / 16.6 ± 0.2 | 0.216 ± 0.139 / 73.3 ± 16.2 / 5.5 ± 0.8 | 0.182 ± 0.160 / 78.9 ± 18.6 / 8.4 ± 0.5 | 0.129 ± 0.117 / 84.4 ± 14.1 / 21.0 ± 12.1 |

Common hits (both WeVisOne gate and YOLO val-cal overlap GT, same seed; n = 125 ± 17 lesions): median HD95 WeVisOne 7.1 ± 0.1 vs YOLO 9.9 ± 0.4 px; Dice 0.864 vs 0.863.

## breast:all

| Model | all (n=252) | benign (n=154) | malignant (n=98) | small <3% (n=93) | medium 3-10% (n=105) | large ≥10% (n=54) |
|---|---|---|---|---|---|---|
| WeVisOne gate | 0.694 ± 0.015 / 14.4 ± 1.6 / 8.1 ± 0.1 | 0.674 ± 0.016 / 16.0 ± 1.6 / 6.5 ± 0.3 | 0.725 ± 0.015 / 11.9 ± 1.6 / 10.3 ± 0.0 | 0.563 ± 0.011 / 26.9 ± 1.9 / 5.9 ± 0.1 | 0.762 ± 0.015 / 8.3 ± 2.4 / 8.0 ± 0.3 | 0.787 ± 0.031 / 4.9 ± 2.1 / 14.6 ± 1.2 |
| YOLO val-cal | 0.335 ± 0.012 / 61.5 ± 1.6 / 7.3 ± 0.7 | 0.334 ± 0.018 / 61.9 ± 2.1 / 5.7 ± 0.3 | 0.335 ± 0.002 / 60.9 ± 1.6 / 8.8 ± 0.9 | 0.233 ± 0.039 / 72.8 ± 4.8 / 4.7 ± 0.5 | 0.341 ± 0.026 / 61.3 ± 2.9 / 6.9 ± 1.0 | 0.496 ± 0.028 / 42.6 ± 4.9 / 9.6 ± 1.0 |
| nnU-Net gate | 0.118 ± 0.108 / 85.6 ± 12.8 / 14.6 ± 12.6 | 0.132 ± 0.121 / 84.2 ± 14.3 / 6.7 ± 0.4 | 0.096 ± 0.090 / 87.8 ± 10.8 / 15.9 ± 11.6 | 0.159 ± 0.131 / 80.3 ± 16.1 / 6.7 ± 1.2 | 0.110 ± 0.114 / 87.0 ± 12.9 / 22.0 ± 24.4 | 0.063 ± 0.090 / 92.0 ± 10.9 / 38.4 ± 38.1 |

Common hits (both WeVisOne gate and YOLO val-cal overlap GT, same seed; n = 97 ± 5 lesions): median HD95 WeVisOne 6.1 ± 0.1 vs YOLO 7.3 ± 0.7 px; Dice 0.871 vs 0.869.
