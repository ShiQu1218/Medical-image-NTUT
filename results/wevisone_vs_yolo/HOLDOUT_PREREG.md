# Pre-registration: hold-out test of the q90 gate (written 2026-11-20, before any hold-out image was downloaded or evaluated)

## Why
The q90 gate was adopted after the original `max` rule failed for 2 of 3 full-ViT seeds on the existing external sets
(post hoc). With 13 BUSI val normals, q90 (numpy linear interpolation, index 0.9 × 12 = 10.8) lies between the
2nd- and 3rd-highest val-normal score (0.8 weight toward the 2nd-highest). This test checks the rule on data never used for any decision.

## Frozen before evaluation (no change allowed after the first hold-out prediction)
- Models: full-ViT `vitfull_s42/43/44` (new main), LLM LoRA `baseline_s42/43/44` (original main), YOLO26m `busi_m_s42/43/44`.
  No retraining, no fine-tuning, no prompt change (`breast lesion`, same prompt as all other cohorts).
- Thresholds: exactly the numbers in `holdout_frozen_thresholds.json` (written together with this file).
  Primary: full-ViT + q90. Secondary (reported, not used for the claim): full-ViT + max, LLM LoRA + max/q90, YOLO26m + val max-conf.
- Hold-out cohort (first that is obtainable, in this order; whichever is used, all its images are used):
  1. QAMEBI (Ardakani et al., 2023; 232 images, benign + malignant, masks)
  2. STU (Shantou University, GE Voluson E10; 42 images, masks)
  Images resized and evaluated exactly as BrEaST (`sources_extra.py` loader pattern, 256×256 metric grid).
- Metrics: lesion miss rate (zero GT overlap), lesion Dice (miss = 0), per pathology if labels exist. FPR only if the cohort has normal images.
- Statistics: mean ± SD over the 3 seeds; paired image-level bootstrap (5,000; images as clusters when no patient ID) for
  full-ViT q90 − LLM LoRA max and full-ViT q90 − YOLO26m.

## Pre-stated success criteria (primary analysis, full-ViT + q90)
1. q90 does not collapse: lesion miss rate ≤ 20% in every seed (the `max` failure mode was ≈ 30%).
2. No worse than the original main method: Δ lesion Dice (full-ViT q90 − LLM LoRA max) 95% CI lower bound > −0.03.
3. Better than YOLO26m: Δ lesion Dice 95% CI lower bound > 0.
All three are reported whatever the outcome; a failure is reported as a failure.
