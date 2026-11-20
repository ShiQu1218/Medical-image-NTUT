Hybrid: WeVisOne decides presence + box (gate = max val-normal rule), fine-tuned MedSAM draws the mask.

Inputs: runs/medsam_ft/<set>_wevisgate_baseline_s{42,43,44}.jsonl (medsam_eval.py --modes wevis:<run>
--decoder runs/medsam_ft/best_decoder.pt). Normal-image FPR is identical to the WeVisOne gate by construction
(same images get boxes); only the mask changes. Compared with WeVisOne gate (own decoder) and YOLO val-cal.
Writes runs/comparison/WEVIS_MEDSAM.md + wevis_medsam.json

| Set | Model | FPR % | lesion Dice | miss % |
|---|---|---:|---:|---:|
| busi:test | WeVisOne gate (own decoder) | 40.0 ± 10.0 | 0.829 ± 0.010 | 1.5 ± 0.0 |
| busi:test | WeVisOne box+gate → MedSAM-ft mask | 40.0 ± 10.0 | 0.828 ± 0.009 | 1.5 ± 0.0 |
| busi:test | YOLO26m val-cal | 0.0 ± 0.0 | 0.704 ± 0.045 | 20.2 ± 5.3 |
| uclm:test | WeVisOne gate (own decoder) | 15.4 ± 6.5 | 0.708 ± 0.032 | 11.0 ± 3.6 |
| uclm:test | WeVisOne box+gate → MedSAM-ft mask | 15.4 ± 6.5 | 0.697 ± 0.031 | 11.2 ± 3.7 |
| uclm:test | YOLO26m val-cal | 9.8 ± 2.7 | 0.506 ± 0.011 | 43.4 ± 0.0 |
| uclm:all | WeVisOne gate (own decoder) | 18.9 ± 8.3 | 0.607 ± 0.023 | 17.4 ± 2.6 |
| uclm:all | WeVisOne box+gate → MedSAM-ft mask | 18.9 ± 8.3 | 0.599 ± 0.023 | 17.4 ± 2.6 |
| uclm:all | YOLO26m val-cal | 12.7 ± 3.1 | 0.380 ± 0.005 | 56.3 ± 1.2 |
| busuc:all | WeVisOne gate (own decoder) | – | 0.879 ± 0.001 | 1.0 ± 0.3 |
| busuc:all | WeVisOne box+gate → MedSAM-ft mask | – | 0.866 ± 0.002 | 1.0 ± 0.3 |
| busuc:all | YOLO26m val-cal | – | 0.502 ± 0.052 | 42.5 ± 6.4 |
| busbra:test | WeVisOne gate (own decoder) | – | 0.732 ± 0.008 | 4.6 ± 0.8 |
| busbra:test | WeVisOne box+gate → MedSAM-ft mask | – | 0.708 ± 0.010 | 3.7 ± 0.9 |
| busbra:test | YOLO26m val-cal | – | 0.291 ± 0.034 | 66.2 ± 4.7 |
| breast:all | WeVisOne gate (own decoder) | 33.3 ± 14.4 | 0.694 ± 0.015 | 14.4 ± 1.6 |
| breast:all | WeVisOne box+gate → MedSAM-ft mask | 33.3 ± 14.4 | 0.680 ± 0.014 | 14.4 ± 1.6 |
| breast:all | YOLO26m val-cal | 16.7 ± 14.4 | 0.335 ± 0.012 | 61.5 ± 1.6 |

## Paired patient-cluster bootstrap (A − B, lesion Dice; Holm over this table)

| Comparison | Δ lesion Dice [95% CI] | Holm p |
|---|---:|---:|
| busi:test / MedSAM-ft hybrid vs WeVisOne gate (own decoder) | -0.001 [-0.006, +0.004] | 0.667 |
| busi:test / MedSAM-ft hybrid vs YOLO26m val-cal | +0.124 [+0.064, +0.184] | 0.0024 |
| uclm:test / MedSAM-ft hybrid vs WeVisOne gate (own decoder) | -0.011 [-0.014, -0.007] | 0.0024 |
| uclm:test / MedSAM-ft hybrid vs YOLO26m val-cal | +0.191 [+0.107, +0.304] | 0.0024 |
| uclm:all / MedSAM-ft hybrid vs WeVisOne gate (own decoder) | -0.008 [-0.011, -0.004] | 0.0024 |
| uclm:all / MedSAM-ft hybrid vs YOLO26m val-cal | +0.219 [+0.161, +0.288] | 0.0024 |
| busuc:all / MedSAM-ft hybrid vs WeVisOne gate (own decoder) | -0.013 [-0.015, -0.011] | 0.0024 |
| busuc:all / MedSAM-ft hybrid vs YOLO26m val-cal | +0.364 [+0.339, +0.389] | 0.0024 |
| busbra:test / MedSAM-ft hybrid vs WeVisOne gate (own decoder) | -0.024 [-0.028, -0.020] | 0.0024 |
| busbra:test / MedSAM-ft hybrid vs YOLO26m val-cal | +0.417 [+0.379, +0.457] | 0.0024 |
| breast:all / MedSAM-ft hybrid vs WeVisOne gate (own decoder) | -0.014 [-0.017, -0.011] | 0.0024 |
| breast:all / MedSAM-ft hybrid vs YOLO26m val-cal | +0.345 [+0.303, +0.388] | 0.0024 |