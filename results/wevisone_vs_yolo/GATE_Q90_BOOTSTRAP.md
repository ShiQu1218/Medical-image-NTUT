Paired patient-cluster bootstrap with the q90 gate (90th percentile of the 13 BUSI val-normal presence scores).

Rule fixed on val only, applied identically to every arm. Cached scores, no inference.
Comparisons: full-ViT vs LLM-only LoRA (BUSI-trained and BUSI+BRA-trained), both q90; plus full-ViT q90 vs the
current main method (LLM-only LoRA, max rule). Holm over every p-value in this table (FPR and lesion Dice).
Writes runs/comparison/GATE_Q90_BOOTSTRAP.md + gate_q90_bootstrap.json

## Descriptive (mean ± SD over seeds 42/43/44)

| Set | arm | rule | FPR % | lesion Dice | miss % |
|---|---|---|---:|---:|---:|
| busi:test | vitfull | q90 | 26.7 ± 20.8 | 0.829 ± 0.004 | 3.0 ± 1.5 |
| busi:test | baseline | q90 | 46.7 ± 5.8 | 0.829 ± 0.010 | 1.5 ± 0.0 |
| busi:test | ms_bra_vitfull | q90 | 16.7 ± 5.8 | 0.815 ± 0.006 | 3.0 ± 1.5 |
| busi:test | ms_bra | q90 | 13.3 ± 5.8 | 0.797 ± 0.005 | 3.0 ± 1.5 |
| busi:test | baseline | max | 40.0 ± 10.0 | 0.829 ± 0.010 | 1.5 ± 0.0 |
| uclm:test | vitfull | q90 | 16.3 ± 6.2 | 0.777 ± 0.044 | 8.6 ± 3.6 |
| uclm:test | baseline | q90 | 16.8 ± 6.6 | 0.712 ± 0.029 | 10.3 ± 2.9 |
| uclm:test | ms_bra_vitfull | q90 | 9.9 ± 3.8 | 0.771 ± 0.056 | 13.1 ± 6.7 |
| uclm:test | ms_bra | q90 | 7.3 ± 1.6 | 0.721 ± 0.031 | 15.2 ± 3.9 |
| uclm:test | baseline | max | 15.4 ± 6.5 | 0.708 ± 0.032 | 11.0 ± 3.6 |
| uclm:all | vitfull | q90 | 21.9 ± 9.0 | 0.687 ± 0.049 | 13.7 ± 4.9 |
| uclm:all | baseline | q90 | 20.4 ± 7.8 | 0.609 ± 0.021 | 16.9 ± 2.0 |
| uclm:all | ms_bra_vitfull | q90 | 11.7 ± 4.4 | 0.697 ± 0.052 | 17.9 ± 6.5 |
| uclm:all | ms_bra | q90 | 9.0 ± 1.5 | 0.632 ± 0.031 | 20.9 ± 3.6 |
| uclm:all | baseline | max | 18.9 ± 8.3 | 0.607 ± 0.023 | 17.4 ± 2.6 |
| busuc:all | vitfull | q90 | – | 0.889 ± 0.005 | 0.4 ± 0.2 |
| busuc:all | baseline | q90 | – | 0.879 ± 0.001 | 0.9 ± 0.2 |
| busuc:all | ms_bra_vitfull | q90 | – | 0.905 ± 0.005 | 0.3 ± 0.2 |
| busuc:all | ms_bra | q90 | – | 0.876 ± 0.007 | 0.6 ± 0.1 |
| busuc:all | baseline | max | – | 0.879 ± 0.001 | 1.0 ± 0.3 |
| busbra:test | vitfull | q90 | – | 0.812 ± 0.007 | 1.4 ± 1.0 |
| busbra:test | baseline | q90 | – | 0.733 ± 0.009 | 4.4 ± 0.7 |
| busbra:test | ms_bra_vitfull | q90 | – | 0.886 ± 0.004 | 0.6 ± 0.2 |
| busbra:test | ms_bra | q90 | – | 0.867 ± 0.004 | 0.3 ± 0.0 |
| busbra:test | baseline | max | – | 0.732 ± 0.008 | 4.6 ± 0.8 |
| breast:all | vitfull | q90 | 50.0 ± 0.0 | 0.682 ± 0.015 | 18.0 ± 2.2 |
| breast:all | baseline | q90 | 41.7 ± 14.4 | 0.699 ± 0.012 | 13.5 ± 1.2 |
| breast:all | ms_bra_vitfull | q90 | 16.7 ± 14.4 | 0.714 ± 0.034 | 15.7 ± 5.0 |
| breast:all | ms_bra | q90 | 33.3 ± 14.4 | 0.684 ± 0.022 | 15.3 ± 3.0 |
| breast:all | baseline | max | 33.3 ± 14.4 | 0.694 ± 0.015 | 14.4 ± 1.6 |

## Paired bootstrap (A − B; 5000 patient-cluster resamples; Holm over this table)

| Comparison | images / patients | Δ FPR pp [95% CI] | Holm p | Δ lesion Dice [95% CI] | Holm p |
|---|---|---:|---:|---:|---:|
| busi:test / full-ViT q90 vs LLM-LoRA q90 | 76 / 76 | -20.0 [-43.3, +0.0] | 1 | -0.001 [-0.022, +0.019] | 1 |
| uclm:test / full-ViT q90 vs LLM-LoRA q90 | 331 / 18 | -0.5 [-5.6, +4.1] | 1 | +0.065 [+0.025, +0.100] | 0.029 |
| uclm:all / full-ViT q90 vs LLM-LoRA q90 | 679 / 38 | +1.5 [-1.9, +4.8] | 1 | +0.078 [+0.051, +0.107] | 0.006 |
| busuc:all / full-ViT q90 vs LLM-LoRA q90 | 811 / 811 | – | – | +0.010 [+0.003, +0.017] | 0.042 |
| busbra:test / full-ViT q90 vs LLM-LoRA q90 | 374 / 212 | – | – | +0.079 [+0.059, +0.100] | 0.006 |
| breast:all / full-ViT q90 vs LLM-LoRA q90 | 256 / 256 | +8.3 [+0.0, +33.3] | 1 | -0.017 [-0.041, +0.004] | 1 |
| busi:test / BUSI+BRA full-ViT q90 vs BUSI+BRA LLM-LoRA q90 | 76 / 76 | +3.3 [-9.5, +20.5] | 1 | +0.018 [-0.018, +0.052] | 1 |
| uclm:test / BUSI+BRA full-ViT q90 vs BUSI+BRA LLM-LoRA q90 | 331 / 18 | +2.7 [-1.7, +6.1] | 1 | +0.050 [+0.005, +0.082] | 0.57 |
| uclm:all / BUSI+BRA full-ViT q90 vs BUSI+BRA LLM-LoRA q90 | 679 / 38 | +2.7 [-0.2, +5.3] | 1 | +0.065 [+0.033, +0.091] | 0.029 |
| busuc:all / BUSI+BRA full-ViT q90 vs BUSI+BRA LLM-LoRA q90 | 811 / 811 | – | – | +0.029 [+0.020, +0.038] | 0.006 |
| busbra:test / BUSI+BRA full-ViT q90 vs BUSI+BRA LLM-LoRA q90 | 374 / 212 | – | – | +0.019 [+0.007, +0.031] | 0.035 |
| breast:all / BUSI+BRA full-ViT q90 vs BUSI+BRA LLM-LoRA q90 | 256 / 256 | -16.7 [-33.3, +0.0] | 1 | +0.030 [+0.002, +0.058] | 0.62 |
| busi:test / full-ViT q90 vs LLM-LoRA max (current main) | 76 / 76 | -13.3 [-33.3, +6.1] | 1 | -0.001 [-0.022, +0.019] | 1 |
| uclm:test / full-ViT q90 vs LLM-LoRA max (current main) | 331 / 18 | +0.9 [-4.1, +5.7] | 1 | +0.069 [+0.029, +0.104] | 0.02 |
| uclm:all / full-ViT q90 vs LLM-LoRA max (current main) | 679 / 38 | +3.0 [-0.2, +6.2] | 1 | +0.081 [+0.053, +0.110] | 0.006 |
| busuc:all / full-ViT q90 vs LLM-LoRA max (current main) | 811 / 811 | – | – | +0.010 [+0.003, +0.018] | 0.042 |
| busbra:test / full-ViT q90 vs LLM-LoRA max (current main) | 374 / 212 | – | – | +0.080 [+0.060, +0.100] | 0.006 |
| breast:all / full-ViT q90 vs LLM-LoRA max (current main) | 256 / 256 | +16.7 [+0.0, +66.7] | 1 | -0.012 [-0.035, +0.010] | 1 |