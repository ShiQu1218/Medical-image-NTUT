# Memorization probe (raw operating point, seed 42 / single model)

If WeVisOne had memorized a cohort (e.g. BUS_UC) during pretraining, its zero-shot drop there should be clearly larger than on the other cohorts. Reports lesion Dice and image-level localized F1 (f1_table.py definition) on the clean and each transformed set, and the relative drop % = 100 * (clean - transformed) / clean. Paired Dice drop uses only images that are lesion images in both versions. Writes runs/comparison/MEMO.md + memo.json. 

| Model | Set | clean Dice | clean F1 | hflip1 Dice drop % | hflip1 F1 drop % | rot1 Dice drop % | rot1 F1 drop % | zoom1 Dice drop % | zoom1 F1 drop % | shift1 Dice drop % | shift1 F1 drop % | mean Dice drop % | mean F1 drop % | paired Dice drop (abs, mean) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WeVisOne zero-shot | busi:test | 0.700 | 0.901 | +2.5 | -1.6 | +6.2 | +0.0 | +1.9 | +1.6 | +1.2 | -1.6 | **+3.0** | **-0.4** | +0.021 |
| WeVisOne zero-shot | uclm:all | 0.554 | 0.526 | -2.4 | -0.4 | +1.5 | +0.4 | -6.1 | -2.0 | -4.2 | +0.4 | **-2.8** | **-0.4** | -0.016 |
| WeVisOne zero-shot | busuc:all | 0.880 | 0.998 | -0.7 | +0.1 | +1.1 | +0.4 | -0.2 | -0.1 | -0.8 | +0.2 | **-0.1** | **+0.2** | -0.000 |
| WeVisOne zero-shot | busbra:test | 0.545 | 0.971 | -3.5 | -0.3 | +14.6 | +1.8 | -0.4 | -0.3 | +2.7 | +1.1 | **+3.4** | **+0.6** | +0.018 |
| WeVisOne zero-shot | breast:all | 0.618 | 0.969 | -2.7 | -2.0 | -0.9 | +0.8 | -3.0 | +0.4 | -5.1 | -0.4 | **-2.9** | **-0.3** | -0.018 |
| WeVisOne fine-tuned s42 | busi:test | 0.833 | 0.949 | +3.9 | +0.1 | +0.7 | +0.0 | +3.7 | -1.4 | +0.9 | -0.7 | **+2.3** | **-0.5** | +0.019 |
| WeVisOne fine-tuned s42 | uclm:all | 0.592 | 0.773 | -1.6 | +2.9 | -5.1 | +5.4 | +0.3 | -4.2 | -4.4 | +6.6 | **-2.7** | **+2.6** | -0.016 |
| WeVisOne fine-tuned s42 | busuc:all | 0.880 | 0.991 | -0.0 | +0.1 | +1.4 | +0.2 | -0.7 | +0.0 | +1.3 | +0.4 | **+0.5** | **+0.2** | +0.005 |
| WeVisOne fine-tuned s42 | busbra:test | 0.741 | 0.957 | +5.1 | +0.3 | +14.3 | +0.0 | +5.2 | -0.7 | +8.1 | -0.3 | **+8.2** | **-0.2** | +0.061 |
| WeVisOne fine-tuned s42 | breast:all | 0.698 | 0.913 | -3.4 | -1.9 | -0.5 | -3.1 | -3.5 | -0.2 | +1.6 | +0.3 | **-1.4** | **-1.2** | -0.010 |
| nnU-Net (BUSI only) | busi:test | 0.823 | 0.978 | -0.1 | +0.0 | +3.6 | +5.5 | +0.7 | +2.2 | +1.1 | +2.2 | **+1.3** | **+2.5** | +0.011 |
| nnU-Net (BUSI only) | uclm:all | 0.493 | 0.489 | +0.1 | +0.0 | +13.3 | +1.5 | -13.8 | -11.9 | -7.5 | -7.1 | **-2.0** | **-4.4** | -0.010 |
| nnU-Net (BUSI only) | busuc:all | 0.819 | 0.992 | -0.1 | -0.1 | +3.5 | -0.2 | -1.5 | -0.4 | +0.1 | -0.6 | **+0.5** | **-0.3** | +0.005 |
| nnU-Net (BUSI only) | busbra:test | 0.677 | 0.928 | -0.0 | +0.3 | +18.6 | +11.4 | +0.0 | +2.4 | -1.0 | +0.8 | **+4.4** | **+3.7** | +0.030 |
| nnU-Net (BUSI only) | breast:all | 0.554 | 0.811 | -0.1 | +0.8 | +14.2 | +9.6 | -3.3 | -1.8 | -3.9 | -0.8 | **+1.7** | **+2.0** | +0.010 |