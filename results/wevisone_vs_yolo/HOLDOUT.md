# Pre-registered hold-out: QAMEBI (232 lesion ROI crops, 109 benign / 123 malignant, no normals)

Frozen files verified unchanged (sha256 HOLDOUT_PREREG.md 5ec365cf6292, holdout_frozen_thresholds.json d9dddea77261).

| Arm | lesion Dice | miss % | miss % per seed | miss benign % | miss malignant % |
|---|---:|---:|---|---:|---:|
| full-ViT q90 (primary) | 0.840 ± 0.011 | 3.0 ± 0.7 | 2.6 / 2.6 / 3.9 | 3.7 | 2.4 |
| full-ViT max | 0.794 ± 0.048 | 8.9 ± 5.4 | 9.1 / 3.4 / 14.2 | 7.3 | 10.3 |
| LLM-LoRA max | 0.827 ± 0.017 | 3.0 ± 1.1 | 4.3 / 2.6 / 2.2 | 1.8 | 4.1 |
| LLM-LoRA q90 | 0.831 ± 0.011 | 2.4 ± 0.2 | 2.6 / 2.6 / 2.2 | 1.8 | 3.0 |
| YOLO26m val-cal | 0.504 ± 0.054 | 41.2 ± 6.3 | 34.1 / 45.7 / 44.0 | 31.8 | 49.6 |

## Paired image-level bootstrap (5,000; seed-averaged per image)

| Comparison | dDice | 95% CI | p |
|---|---:|---|---:|
| full-ViT q90 (primary) - LLM-LoRA max | +0.013 | [-0.001, +0.028] | 0.0656 |
| full-ViT q90 (primary) - YOLO26m val-cal | +0.336 | [+0.290, +0.381] | 0.0000 |

## Pre-registered success criteria

- 1. miss <= 20% in every seed: **PASS**
- 2. dDice vs LLM-LoRA max: CI low > -0.03: **PASS**
- 3. dDice vs YOLO26m: CI low > 0: **PASS**

Secondary: LLM-LoRA max - YOLO26m dDice +0.323 [+0.278, +0.367].