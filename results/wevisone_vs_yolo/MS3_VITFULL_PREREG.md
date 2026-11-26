# Pre-registration: full-ViT fine-tune trained on BUSI + BUS-BRA + UCLM train (written 2026-11-21, before any training)

## Question
With the LLM-LoRA method, adding UCLM train patients reversed the comparison on UCLM test (WeVisOne per-source gate
0.641 vs YOLO26m 0.734, dDice -0.093 [-0.145, -0.033]). Does the full-ViT method tie or beat YOLO26m in the same regime?

## Prediction (stated before training)
Point estimate of dDice (full-ViT - YOLO26m) on UCLM test between -0.05 and +0.03, i.e. roughly a tie, no clear win.
Reason: full-ViT adds about +0.06 to +0.07 over LLM-LoRA in the BUSI and BUSI+BRA regimes; 0.641 + 0.065 ~ 0.71 vs 0.734.

## Frozen protocol
- Runs: `ms_bra_uclm_vitfull_s42/43/44`. Identical to the local `ms_bra_uclm_s*` LoRA runs except the ViT:
  `--vit_full --lr_vit 1e-5 --aug strong --train_spec busi:train+busbra:train+uclm:train
  --val_spec busi:val+busbra:val+uclm:val --total_steps 690 --n_vals 4 --seed <s>`. Same model revision (sha256 checked).
- Eval sets: busi:val+busbra:val+uclm:val, busi:test, uclm:all (uclm:test derived by patient filter), busuc:all,
  busbra:test, breast:all. UCLM test patients, BUS_UC and BrEaST are never used for training or selection.
- Gate (primary): per-source q90 for BOTH arms. A test cohort uses the q90 of its own source's val normals when that
  source has normals in the pooled val (UCLM: 60 normals), otherwise the pooled val normals. YOLO26m (`ms_bra_uclm_s*`,
  already trained) uses the same rule on its max detection confidence. Secondary: per-source max for both arms
  (the rule used for the earlier LoRA result), and ungated (raw).
- Metric: lesion Dice on the 256x256 grid, misses = 0; also miss rate and normal-image FPR.
- Statistics: patient-cluster paired bootstrap (5,000, seed 2026) of seed-averaged per-image Dice, as `analyze.paired_bootstrap`.

## Primary endpoint and decision rule (UCLM test, full-ViT - YOLO26m, per-source q90)
- 95% CI lower bound > 0: full-ViT **wins**.
- 95% CI upper bound < 0: full-ViT **loses**.
- 95% CI entirely within [-0.05, +0.05]: **equivalent** (tie within the margin).
- Otherwise: **inconclusive** (CI includes 0 but is wider than the margin).
All outcomes are reported, whatever they are.

## Secondary (reported, not used for the primary claim)
1. full-ViT - LLM-LoRA (`ms_bra_uclm_s*`) on UCLM test, same rule.
2. full-ViT - YOLO26m on cohorts never seen in training: busuc:all, breast:all.
3. Same comparisons with per-source max and ungated.

## Exploratory (hypothesis formed after seeing the LoRA result; never a confirmatory claim)
Hypothesis: adding UCLM train makes the model answer "None" too often on UCLM-looking images, because UCLM train is
65% normal (171 / 262) while the pooled train set is 11% normal. Test: `ms_bra_uclm_vitfull_bal_s42/43/44`, identical to the
primary runs plus `--src_lesion_repeat uclm:3` (UCLM lesion images x3 in the epoch pool, fixed update budget; UCLM share
of normals in an epoch 65% -> about 38%). Prediction: lower UCLM test miss rate than the primary runs, with higher FPR.
