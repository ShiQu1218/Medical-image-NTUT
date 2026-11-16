# Paired Dice on commonly hit lesion images

A = WeVisOne baseline, B = YOLO26m-seg. Pair the same seed and identical image IDs. Report both-hit, only-A, only-B and neither-hit partitions for each criterion. Dice is averaged ONLY on both-hit lesion images within each seed, then summarized as mean ± sample SD across seeds. Counts pooled across seeds repeat images; per-seed IDs and Dice are in common_hits.json. The selected cohort depends on both models and the hit criterion; this conditional comparison cannot establish an unconditional segmentation advantage or isolate decoder quality. Use the all-lesion and GT-box oracle results alongside it. All score thresholds are validation-fixed/default, with no test-set retuning.

| Set | A / B op | criterion | runs | both / only A / only B / neither (seed-pooled) | A Dice | B Dice | paired ΔDice |
|---|---|---|---:|---|---:|---:|---:|
| busi:test | raw / 0.25 | any_overlap | 3 | 184 / 11 / 0 / 3 | 0.848 ± 0.015 | 0.834 ± 0.026 | 0.014 ± 0.013 |
| busi:test | raw / 0.25 | iou_0.25 | 3 | 171 / 14 / 1 / 12 | 0.888 ± 0.006 | 0.886 ± 0.012 | 0.002 ± 0.007 |
| busi:test | raw / 0.25 | iou_0.50 | 3 | 159 / 18 / 4 / 17 | 0.903 ± 0.003 | 0.904 ± 0.002 | -0.001 ± 0.003 |
| busi:test | gate / cal | any_overlap | 3 | 158 / 37 / 0 / 3 | 0.877 ± 0.004 | 0.883 ± 0.003 | -0.006 ± 0.002 |
| busi:test | gate / cal | iou_0.25 | 3 | 153 / 32 / 0 / 13 | 0.901 ± 0.008 | 0.910 ± 0.012 | -0.009 ± 0.005 |
| busi:test | gate / cal | iou_0.50 | 3 | 147 / 30 / 4 / 17 | 0.911 ± 0.003 | 0.916 ± 0.006 | -0.005 ± 0.003 |
| uclm:test | raw / 0.25 | any_overlap | 3 | 282 / 105 / 19 / 23 | 0.845 ± 0.019 | 0.837 ± 0.019 | 0.008 ± 0.027 |
| uclm:test | raw / 0.25 | iou_0.25 | 3 | 250 / 88 / 36 / 55 | 0.917 ± 0.009 | 0.877 ± 0.019 | 0.040 ± 0.026 |
| uclm:test | raw / 0.25 | iou_0.50 | 3 | 233 / 89 / 27 / 80 | 0.925 ± 0.004 | 0.897 ± 0.015 | 0.028 ± 0.019 |
| uclm:test | gate / cal | any_overlap | 3 | 231 / 151 / 12 / 35 | 0.881 ± 0.026 | 0.896 ± 0.024 | -0.014 ± 0.014 |
| uclm:test | gate / cal | iou_0.25 | 3 | 215 / 119 / 24 / 71 | 0.927 ± 0.005 | 0.912 ± 0.010 | 0.014 ± 0.005 |
| uclm:test | gate / cal | iou_0.50 | 3 | 212 / 106 / 21 / 90 | 0.929 ± 0.005 | 0.916 ± 0.010 | 0.012 ± 0.006 |
| uclm:all | raw / 0.25 | any_overlap | 3 | 430 / 220 / 34 / 96 | 0.788 ± 0.021 | 0.809 ± 0.023 | -0.022 ± 0.016 |
| uclm:all | raw / 0.25 | iou_0.25 | 3 | 363 / 177 / 72 / 168 | 0.886 ± 0.009 | 0.857 ± 0.025 | 0.029 ± 0.029 |
| uclm:all | raw / 0.25 | iou_0.50 | 3 | 320 / 168 / 59 / 233 | 0.908 ± 0.006 | 0.886 ± 0.015 | 0.022 ± 0.020 |
| uclm:all | gate / cal | any_overlap | 3 | 324 / 320 / 17 / 119 | 0.829 ± 0.029 | 0.872 ± 0.022 | -0.042 ± 0.012 |
| uclm:all | gate / cal | iou_0.25 | 3 | 289 / 247 / 42 / 202 | 0.902 ± 0.009 | 0.897 ± 0.016 | 0.005 ± 0.008 |
| uclm:all | gate / cal | iou_0.50 | 3 | 275 / 209 / 42 / 254 | 0.914 ± 0.005 | 0.908 ± 0.010 | 0.006 ± 0.005 |
| busuc:all | raw / 0.25 | any_overlap | 3 | 1955 / 456 / 5 / 17 | 0.902 ± 0.004 | 0.782 ± 0.061 | 0.120 ± 0.057 |
| busuc:all | raw / 0.25 | iou_0.25 | 3 | 1781 / 582 / 14 / 56 | 0.912 ± 0.002 | 0.830 ± 0.035 | 0.082 ± 0.033 |
| busuc:all | raw / 0.25 | iou_0.50 | 3 | 1525 / 768 / 32 / 108 | 0.923 ± 0.001 | 0.873 ± 0.012 | 0.050 ± 0.012 |
| busuc:all | gate / cal | any_overlap | 3 | 1398 / 1010 / 1 / 24 | 0.920 ± 0.004 | 0.873 ± 0.006 | 0.047 ± 0.002 |
| busuc:all | gate / cal | iou_0.25 | 3 | 1376 / 986 / 5 / 66 | 0.923 ± 0.002 | 0.882 ± 0.006 | 0.041 ± 0.006 |
| busuc:all | gate / cal | iou_0.50 | 3 | 1332 / 961 / 15 / 125 | 0.927 ± 0.002 | 0.889 ± 0.005 | 0.038 ± 0.006 |
| busbra:test | raw / 0.25 | any_overlap | 3 | 569 / 504 / 6 / 43 | 0.835 ± 0.027 | 0.797 ± 0.080 | 0.038 ± 0.054 |
| busbra:test | raw / 0.25 | iou_0.25 | 3 | 498 / 479 / 24 / 121 | 0.867 ± 0.006 | 0.842 ± 0.033 | 0.025 ± 0.028 |
| busbra:test | raw / 0.25 | iou_0.50 | 3 | 412 / 418 / 48 / 244 | 0.900 ± 0.003 | 0.874 ± 0.012 | 0.026 ± 0.009 |
| busbra:test | gate / cal | any_overlap | 3 | 376 / 694 / 3 / 49 | 0.864 ± 0.021 | 0.863 ± 0.022 | 0.001 ± 0.001 |
| busbra:test | gate / cal | iou_0.25 | 3 | 367 / 609 / 11 / 135 | 0.877 ± 0.012 | 0.866 ± 0.018 | 0.012 ± 0.007 |
| busbra:test | gate / cal | iou_0.50 | 3 | 321 / 508 / 30 / 263 | 0.907 ± 0.002 | 0.888 ± 0.006 | 0.019 ± 0.005 |
| breast:all | raw / 0.25 | any_overlap | 3 | 403 / 253 / 11 / 89 | 0.845 ± 0.016 | 0.821 ± 0.025 | 0.024 ± 0.018 |
| breast:all | raw / 0.25 | iou_0.25 | 3 | 365 / 233 / 22 / 136 | 0.890 ± 0.009 | 0.860 ± 0.009 | 0.030 ± 0.001 |
| breast:all | raw / 0.25 | iou_0.50 | 3 | 328 / 228 / 20 / 180 | 0.902 ± 0.004 | 0.889 ± 0.007 | 0.012 ± 0.004 |
| breast:all | gate / cal | any_overlap | 3 | 290 / 357 / 1 / 108 | 0.871 ± 0.009 | 0.869 ± 0.016 | 0.002 ± 0.012 |
| breast:all | gate / cal | iou_0.25 | 3 | 275 / 318 / 9 / 154 | 0.906 ± 0.003 | 0.889 ± 0.010 | 0.016 ± 0.009 |
| breast:all | gate / cal | iou_0.50 | 3 | 262 / 289 / 7 / 198 | 0.911 ± 0.003 | 0.904 ± 0.006 | 0.007 ± 0.007 |