# Hybrid: WeVisOne localization → YOLO mask

Cached predictions only, no inference. Pairs WeVisOne baseline_s{s} (gate) with YOLO busi_m_s{s}, s = 42/43/44.
YOLO caches every detection with conf >= 0.001 (max 20), with masks on the 256 grid.
Rule (prespecified, nothing tuned on test): split the gated WeVisOne mask into connected components; for each
component take the YOLO detection with the HIGHEST confidence among those whose mask has IoU >= 0.1 with the
component, and use that YOLO mask instead of the component.
  (a) W-loc -> YOLO mask, fallback WeVisOne : no matching YOLO detection -> keep the WeVisOne component
  (b) W-loc -> YOLO mask, no fallback       : no matching YOLO detection -> drop the component
  (c) (a) + YOLO val-cal detections that do not overlap any WeVisOne component (union of detectors)
Normal-image FPR follows the WeVisOne gate in (a)/(b); (c) also inherits YOLO-cal false positives.
Writes runs/comparison/HYBRID_WLOC_YSEG.md + hybrid_wloc_yseg.json

Cells: mean ± SD over seeds 42/43/44. HD95 = median over lesion images with a nonempty mask (256-grid px).

## busi:test

Share of WeVisOne components replaced by a YOLO mask: 94.5%

| Model | normal FPR % | lesion Dice | miss % | HD95 px |
|---|---:|---:|---:|---:|
| WeVisOne gate (reference) | 40.0 ± 10.0 | 0.829 ± 0.010 | 1.5 ± 0.0 | – |
| YOLO val-cal (reference) | 0.0 ± 0.0 | 0.704 ± 0.045 | 20.2 ± 5.3 | – |
| (a) W-loc → YOLO mask, fallback W | 40.0 ± 10.0 | 0.829 ± 0.001 | 1.5 ± 0.0 | 7.239 ± 1.154 |
| (b) W-loc → YOLO mask, no fallback | 33.3 ± 11.5 | 0.829 ± 0.001 | 1.5 ± 0.0 | 7.014 ± 1.098 |
| (c) (a) ∪ YOLO-cal detections | 40.0 ± 10.0 | 0.824 ± 0.006 | 1.5 ± 0.0 | 7.468 ± 1.361 |

## uclm:test

Share of WeVisOne components replaced by a YOLO mask: 77.2%

| Model | normal FPR % | lesion Dice | miss % | HD95 px |
|---|---:|---:|---:|---:|
| WeVisOne gate (reference) | 15.4 ± 6.5 | 0.708 ± 0.032 | 11.0 ± 3.6 | – |
| YOLO val-cal (reference) | 9.8 ± 2.7 | 0.506 ± 0.011 | 43.4 ± 0.0 | – |
| (a) W-loc → YOLO mask, fallback W | 15.4 ± 6.5 | 0.696 ± 0.040 | 11.2 ± 3.2 | 9.744 ± 1.094 |
| (b) W-loc → YOLO mask, no fallback | 12.8 ± 6.5 | 0.672 ± 0.049 | 18.6 ± 5.3 | 8.633 ± 1.319 |
| (c) (a) ∪ YOLO-cal detections | 22.9 ± 3.8 | 0.716 ± 0.051 | 8.4 ± 4.4 | 10.685 ± 1.883 |

## uclm:all

Share of WeVisOne components replaced by a YOLO mask: 68.1%

| Model | normal FPR % | lesion Dice | miss % | HD95 px |
|---|---:|---:|---:|---:|
| WeVisOne gate (reference) | 18.9 ± 8.3 | 0.607 ± 0.023 | 17.4 ± 2.6 | – |
| YOLO val-cal (reference) | 12.7 ± 3.1 | 0.380 ± 0.005 | 56.3 ± 1.2 | – |
| (a) W-loc → YOLO mask, fallback W | 18.9 ± 8.3 | 0.596 ± 0.020 | 17.8 ± 2.5 | 17.698 ± 2.904 |
| (b) W-loc → YOLO mask, no fallback | 16.5 ± 8.2 | 0.560 ± 0.033 | 28.5 ± 4.8 | 13.196 ± 1.055 |
| (c) (a) ∪ YOLO-cal detections | 28.2 ± 4.7 | 0.610 ± 0.027 | 15.6 ± 3.1 | 19.210 ± 3.127 |

## busuc:all

Share of WeVisOne components replaced by a YOLO mask: 91.2%

| Model | normal FPR % | lesion Dice | miss % | HD95 px |
|---|---:|---:|---:|---:|
| WeVisOne gate (reference) | – | 0.879 ± 0.001 | 1.0 ± 0.3 | – |
| YOLO val-cal (reference) | – | 0.502 ± 0.052 | 42.5 ± 6.4 | – |
| (a) W-loc → YOLO mask, fallback W | – | 0.767 ± 0.002 | 1.1 ± 0.3 | 20.375 ± 2.713 |
| (b) W-loc → YOLO mask, no fallback | – | 0.750 ± 0.018 | 3.3 ± 2.6 | 20.640 ± 2.361 |
| (c) (a) ∪ YOLO-cal detections | – | 0.765 ± 0.003 | 1.0 ± 0.2 | 21.027 ± 2.641 |

## busbra:test

Share of WeVisOne components replaced by a YOLO mask: 77.9%

| Model | normal FPR % | lesion Dice | miss % | HD95 px |
|---|---:|---:|---:|---:|
| WeVisOne gate (reference) | – | 0.732 ± 0.008 | 4.6 ± 0.8 | – |
| YOLO val-cal (reference) | – | 0.291 ± 0.034 | 66.2 ± 4.7 | – |
| (a) W-loc → YOLO mask, fallback W | – | 0.671 ± 0.026 | 5.4 ± 0.8 | 21.905 ± 2.693 |
| (b) W-loc → YOLO mask, no fallback | – | 0.646 ± 0.042 | 12.6 ± 3.5 | 19.850 ± 1.753 |
| (c) (a) ∪ YOLO-cal detections | – | 0.670 ± 0.027 | 5.3 ± 0.7 | 22.057 ± 2.838 |

## breast:all

Share of WeVisOne components replaced by a YOLO mask: 88.0%

| Model | normal FPR % | lesion Dice | miss % | HD95 px |
|---|---:|---:|---:|---:|
| WeVisOne gate (reference) | 33.3 ± 14.4 | 0.694 ± 0.015 | 14.4 ± 1.6 | – |
| YOLO val-cal (reference) | 16.7 ± 14.4 | 0.335 ± 0.012 | 61.5 ± 1.6 | – |
| (a) W-loc → YOLO mask, fallback W | 33.3 ± 14.4 | 0.650 ± 0.014 | 14.6 ± 1.7 | 12.542 ± 1.802 |
| (b) W-loc → YOLO mask, no fallback | 33.3 ± 14.4 | 0.628 ± 0.029 | 17.9 ± 3.8 | 12.243 ± 1.565 |
| (c) (a) ∪ YOLO-cal detections | 50.0 ± 0.0 | 0.650 ± 0.014 | 14.4 ± 1.9 | 12.690 ± 1.559 |

## Paired patient-cluster bootstrap (A − B; seed-averaged; Holm over this table)

| Comparison | ΔFPR pp [95% CI] | Δ lesion Dice [95% CI] | Dice Holm p |
|---|---:|---:|---:|
| uclm:test / (a) W-loc → YOLO mask, fallback W vs WeVisOne gate | +0.0 [+0.0, +0.0] | -0.012 [-0.031, +0.003] | 0.792 |
| uclm:test / (a) W-loc → YOLO mask, fallback W vs YOLO val-cal | +5.7 [-0.3, +12.5] | +0.190 [+0.111, +0.292] | 0.0036 |
| uclm:test / (b) W-loc → YOLO mask, no fallback vs WeVisOne gate | -2.7 [-7.0, -0.3] | -0.036 [-0.074, -0.009] | 0.048 |
| uclm:test / (b) W-loc → YOLO mask, no fallback vs YOLO val-cal | +3.0 [-1.9, +8.3] | +0.166 [+0.096, +0.259] | 0.0036 |
| uclm:test / (c) (a) ∪ YOLO-cal detections vs WeVisOne gate | +7.4 [+4.7, +10.4] | +0.008 [-0.022, +0.037] | 1 |
| uclm:test / (c) (a) ∪ YOLO-cal detections vs YOLO val-cal | +13.1 [+8.2, +18.9] | +0.210 [+0.137, +0.308] | 0.0036 |
| uclm:all / (a) W-loc → YOLO mask, fallback W vs WeVisOne gate | +0.0 [+0.0, +0.0] | -0.011 [-0.027, +0.005] | 1 |
| uclm:all / (a) W-loc → YOLO mask, fallback W vs YOLO val-cal | +6.1 [+1.3, +11.1] | +0.216 [+0.161, +0.280] | 0.0036 |
| uclm:all / (b) W-loc → YOLO mask, no fallback vs WeVisOne gate | -2.4 [-4.6, -0.9] | -0.047 [-0.076, -0.020] | 0.0036 |
| uclm:all / (b) W-loc → YOLO mask, no fallback vs YOLO val-cal | +3.7 [-0.4, +7.7] | +0.180 [+0.134, +0.235] | 0.0036 |
| uclm:all / (c) (a) ∪ YOLO-cal detections vs WeVisOne gate | +9.3 [+6.8, +12.1] | +0.003 [-0.016, +0.024] | 1 |
| uclm:all / (c) (a) ∪ YOLO-cal detections vs YOLO val-cal | +15.4 [+11.8, +19.6] | +0.230 [+0.179, +0.291] | 0.0036 |
| busi:test / (a) W-loc → YOLO mask, fallback W vs WeVisOne gate | +0.0 [+0.0, +0.0] | -0.000 [-0.015, +0.017] | 1 |
| busi:test / (a) W-loc → YOLO mask, fallback W vs YOLO val-cal | +40.0 [+13.3, +66.7] | +0.125 [+0.071, +0.180] | 0.0036 |
| busi:test / (b) W-loc → YOLO mask, no fallback vs WeVisOne gate | -6.7 [-22.2, +0.0] | -0.000 [-0.015, +0.017] | 1 |
| busi:test / (b) W-loc → YOLO mask, no fallback vs YOLO val-cal | +33.3 [+7.4, +63.0] | +0.125 [+0.071, +0.180] | 0.0036 |
| busi:test / (c) (a) ∪ YOLO-cal detections vs WeVisOne gate | +0.0 [+0.0, +0.0] | -0.006 [-0.021, +0.011] | 1 |
| busi:test / (c) (a) ∪ YOLO-cal detections vs YOLO val-cal | +40.0 [+13.3, +66.7] | +0.120 [+0.066, +0.175] | 0.0036 |