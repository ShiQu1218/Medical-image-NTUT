# nnU-Net trained on BUS-BRA only (raw, no gate)

| Test set | n | normals | FPR % | lesion Dice | miss % | image-level F1 |
|---|---:|---:|---:|---:|---:|---:|
| BUS-BRA val (in-domain) | 195 | 0 | – | 0.912 | 0.0 | 1.000 |
| BUS-BRA test (in-domain) | 374 | 0 | – | 0.892 | 0.5 | 0.996 |
| BUSI uncleaned lesions 647 (external) | 647 | 0 | – | 0.635 | 7.0 | 0.941 |
| UCLM all (external) | 679 | 419 | 99.8 | 0.534 | 3.1 | 0.537 |