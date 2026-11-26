# BiomedParse v1 (zero-shot, text prompt, no box)

BUSI val-normal gate thresholds on max-prob: max=0.999951, q90=0.999925.

| Set | op | lesion Dice | miss % | FPR % | AUROC |
|---|---|---:|---:|---:|---:|
| busi:test | raw | 0.894 | 1.5 | 100.0 | 0.959 |
| busi:test | max | 0.812 | 12.1 | 0.0 | 0.959 |
| busi:test | q90 | 0.812 | 12.1 | 0.0 | 0.959 |
| uclm:test | raw | 0.730 | 8.4 | 99.5 | 0.790 |
| uclm:test | max | 0.574 | 34.3 | 26.6 | 0.790 |
| uclm:test | q90 | 0.595 | 31.5 | 29.8 | 0.790 |
| uclm:all | raw | 0.644 | 7.7 | 99.8 | 0.688 |
| uclm:all | max | 0.470 | 43.1 | 33.2 | 0.688 |
| uclm:all | q90 | 0.484 | 40.8 | 37.2 | 0.688 |
| busuc:all | raw | 0.843 | 1.5 | — | — |
| busuc:all | max | 0.578 | 35.4 | — | — |
| busuc:all | q90 | 0.615 | 31.1 | — | — |
| busbra:test | raw | 0.769 | 2.7 | — | — |
| busbra:test | max | 0.476 | 43.9 | — | — |
| busbra:test | q90 | 0.515 | 39.0 | — | — |
| breast:all | raw | 0.734 | 4.8 | 100.0 | 0.358 |
| breast:all | max | 0.453 | 46.8 | 75.0 | 0.358 |
| breast:all | q90 | 0.478 | 43.3 | 75.0 | 0.358 |
| qamebi:all | raw | 0.845 | 0.4 | — | — |
| qamebi:all | max | 0.698 | 20.7 | — | — |
| qamebi:all | q90 | 0.717 | 18.5 | — | — |

## Paired patient-cluster bootstrap (BiomedParse max-gate as B)

| Set | A | dDice (A-B) | 95% CI | dFPR | 95% CI |
|---|---|---:|---|---:|---|
| busi:test | full-ViT q90 | +0.017 | [-0.046, +0.083] | +0.267 | [+0.077, +0.500] |
| busi:test | YOLO26m cal | -0.108 | [-0.190, -0.021] | +0.000 | [+0.000, +0.000] |
| uclm:test | full-ViT q90 | +0.203 | [+0.118, +0.308] | -0.103 | [-0.175, -0.042] |
| uclm:test | YOLO26m cal | -0.068 | [-0.165, +0.011] | -0.168 | [-0.252, -0.099] |
| uclm:all | full-ViT q90 | +0.217 | [+0.150, +0.290] | -0.113 | [-0.188, -0.041] |
| uclm:all | YOLO26m cal | -0.090 | [-0.166, -0.023] | -0.204 | [-0.272, -0.140] |
| busuc:all | full-ViT q90 | +0.311 | [+0.284, +0.340] | +nan | [+nan, +nan] |
| busuc:all | YOLO26m cal | -0.077 | [-0.107, -0.045] | +nan | [+nan, +nan] |
| busbra:test | full-ViT q90 | +0.336 | [+0.291, +0.381] | +nan | [+nan, +nan] |
| busbra:test | YOLO26m cal | -0.185 | [-0.235, -0.136] | +nan | [+nan, +nan] |
| breast:all | full-ViT q90 | +0.229 | [+0.181, +0.278] | -0.250 | [-1.000, +0.000] |
| breast:all | YOLO26m cal | -0.118 | [-0.169, -0.070] | -0.583 | [-1.000, +0.000] |
| qamebi:all | YOLO26m cal | -0.195 | [-0.241, -0.149] | +nan | [+nan, +nan] |