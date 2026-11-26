# BrEaST error analysis (new main method: full-ViT + q90; original: LLM LoRA + max)

Lesion images: 252 (154 benign / 98 malignant). Miss = zero GT overlap after the val-fitted gate. Per-image miss rates are averaged over seeds 42/43/44; 'consistent miss' = missed in >= 2 of 3 seeds.

## 1. Miss types (lesion-image x seed counts)

| Arm | misses | gate (says no lesion) | no mask | wrong location |
|---|---:|---:|---:|---:|
| full-ViT q90 | 136 (18.0%) | 112 | 11 | 13 |
| LLM-LoRA max | 109 (14.4%) | 81 | 8 | 20 |
| LLM-LoRA q90 | 102 (13.5%) | 69 | 12 | 21 |

## 2. Gate threshold vs lesion presence scores

| Arm | thresholds (s42/43/44) | median lesion score BrEaST | median lesion score other cohorts |
|---|---|---:|---:|
| full-ViT q90 | 0.00 / -0.67 / 0.08 | 4.00 | busi 5.97, uclm 4.88, busuc 7.00, busbra 7.25 |
| LLM-LoRA max | 1.12 / -0.75 / 0.12 | 6.19 | busi 6.88, uclm 5.62, busuc 7.19, busbra 6.88 |
| LLM-LoRA q90 | 0.60 / -1.25 / 0.00 | 6.19 | busi 6.88, uclm 5.62, busuc 7.19, busbra 6.88 |

## 3. Lesion size: BrEaST vs other cohorts (GT area fraction of the image)

| Cohort | n lesions | median area % | share < 1% | share < 3% |
|---|---:|---:|---:|---:|
| busi:test | 66 | 5.23 | 8% | 33% |
| uclm:test | 143 | 5.34 | 10% | 32% |
| busuc:all | 810 | 11.64 | 0% | 7% |
| busbra:test | 374 | 6.80 | 3% | 23% |
| breast:all | 252 | 4.90 | 4% | 37% |

BrEaST physical size: median equivalent diameter 10.2 mm (IQR 6.7-14.2); lesions < 5 mm: 20, < 10 mm: 124. Field of view width median 47 mm; image width median 545 px (min 363, max 1024).

### Pathology

| Pathology | n | miss % full-ViT q90 | miss % LLM-LoRA max | miss % LLM-LoRA q90 |
|---|---:|---:|---:|---:|
| benign | 154 | 21.6 | 16.0 | 15.2 |
| malignant | 98 | 12.2 | 11.9 | 10.9 |

### Equivalent diameter

| Equivalent diameter | n | miss % full-ViT q90 | miss % LLM-LoRA max | miss % LLM-LoRA q90 |
|---|---:|---:|---:|---:|
| <5 mm | 20 | 48.3 | 51.7 | 48.3 |
| 5-10 mm | 104 | 21.5 | 15.4 | 14.4 |
| 10-15 mm | 74 | 13.5 | 11.7 | 10.8 |
| >=20 mm | 26 | 7.7 | 5.1 | 5.1 |
| 15-20 mm | 28 | 4.8 | 0.0 | 0.0 |

### Image width

| Image width | n | miss % full-ViT q90 | miss % LLM-LoRA max | miss % LLM-LoRA q90 |
|---|---:|---:|---:|---:|
| 600-800 px | 73 | 19.6 | 14.6 | 13.7 |
| >=800 px | 40 | 18.3 | 14.2 | 12.5 |
| width <600 px | 139 | 17.0 | 14.4 | 13.7 |

### BI-RADS

| BI-RADS | n | miss % full-ViT q90 | miss % LLM-LoRA max | miss % LLM-LoRA q90 |
|---|---:|---:|---:|---:|
| 2 | 30 | 23.3 | 10.0 | 8.9 |
| 3 | 37 | 21.6 | 18.0 | 17.1 |
| 4a | 44 | 21.2 | 18.9 | 17.4 |
| 4b | 46 | 15.9 | 13.8 | 13.8 |
| 4c | 49 | 15.6 | 11.6 | 10.2 |
| 5 | 46 | 13.0 | 13.8 | 13.0 |

### Shape

| Shape | n | miss % full-ViT q90 | miss % LLM-LoRA max | miss % LLM-LoRA q90 |
|---|---:|---:|---:|---:|
| round | 15 | 28.9 | 24.4 | 22.2 |
| oval | 97 | 19.6 | 14.8 | 14.1 |
| irregular | 140 | 15.7 | 13.1 | 12.1 |

### Margin

| Margin | n | miss % full-ViT q90 | miss % LLM-LoRA max | miss % LLM-LoRA q90 |
|---|---:|---:|---:|---:|
| not circumscribed - indistinct | 52 | 27.6 | 20.5 | 19.9 |
| circumscribed | 115 | 22.3 | 17.4 | 16.2 |
| not circumscribed - angular&indistinct | 23 | 8.7 | 8.7 | 8.7 |
| not circumscribed - spiculated&indistinct | 11 | 3.0 | 3.0 | 3.0 |
| not circumscribed - microlobulated | 10 | 0.0 | 0.0 | 0.0 |
| not circumscribed - microlobulated&indistinct | 11 | 0.0 | 3.0 | 0.0 |

### Echogenicity

| Echogenicity | n | miss % full-ViT q90 | miss % LLM-LoRA max | miss % LLM-LoRA q90 |
|---|---:|---:|---:|---:|
| isoechoic | 12 | 61.1 | 58.3 | 52.8 |
| hyperechoic | 9 | 44.4 | 14.8 | 14.8 |
| heterogeneous | 57 | 15.8 | 12.3 | 11.1 |
| hypoechoic | 148 | 15.5 | 13.3 | 12.6 |
| complex cystic/solid | 11 | 9.1 | 9.1 | 9.1 |
| anechoic | 15 | 6.7 | 2.2 | 2.2 |

### Posterior features

| Posterior features | n | miss % full-ViT q90 | miss % LLM-LoRA max | miss % LLM-LoRA q90 |
|---|---:|---:|---:|---:|
| shadowing | 50 | 23.3 | 17.3 | 14.7 |
| no | 159 | 20.3 | 17.2 | 16.6 |
| enhancement | 36 | 3.7 | 0.9 | 0.9 |

### Tissue composition

| Tissue composition | n | miss % full-ViT q90 | miss % LLM-LoRA max | miss % LLM-LoRA q90 |
|---|---:|---:|---:|---:|
| heterogeneous: predominantly fat | 78 | 24.4 | 20.5 | 19.7 |
| homogeneous: fibroglandular | 44 | 21.2 | 16.7 | 15.2 |
| homogeneous: fat | 24 | 18.1 | 16.7 | 16.7 |
| not available | 49 | 15.0 | 10.9 | 10.2 |
| heterogeneous: predominantly fibroglandular | 56 | 9.5 | 6.5 | 5.4 |

### Interpretation group

| Interpretation group | n | miss % full-ViT q90 | miss % LLM-LoRA max | miss % LLM-LoRA q90 |
|---|---:|---:|---:|---:|
| duct-related | 21 | 23.8 | 20.6 | 20.6 |
| other | 85 | 18.0 | 13.7 | 12.5 |
| fibroadenoma | 101 | 17.2 | 14.9 | 14.2 |
| cyst | 28 | 13.1 | 11.9 | 11.9 |

### GT components

| GT components | n | miss % full-ViT q90 | miss % LLM-LoRA max | miss % LLM-LoRA q90 |
|---|---:|---:|---:|---:|
| 1 | 252 | 18.0 | 14.4 | 13.5 |

## 5. Consistently missed lesions (full-ViT q90, missed in >= 2/3 seeds)

n = 44 of 252. Median diameter 6.5 mm vs 10.9 mm for the rest; malignant 23% vs 42%; median image width 584 vs 545 px.

| id | class | BI-RADS | diam mm | area % | width px | shape | margin | echo | posterior | interpretation | miss types (s42/43/44) | LoRA-max missed |
|---|---|---|---:|---:|---:|---|---|---|---|---|---|---:|
| case177.png | benign | 3 | 3.0 | 0.62 | 794 | round | circumscribed | hypoechoic | no | Cyst filled with thick fluid | gate/gate/gate | 1.00 |
| case033.png | benign | 4a | 3.3 | 0.63 | 794 | oval | circumscribed | isoechoic | no | Suspicion of malignancy&Mammary duct ectasia&Intraductal pap | gate/gate/gate | 1.00 |
| case181.png | benign | 3 | 3.4 | 0.58 | 485 | oval | circumscribed | hypoechoic | no | Complex cyst / Non-simple cyst&Cyst filled with thick fluid& | gate/hit/gate | 1.00 |
| case199.png | malignant | 4c | 3.4 | 0.58 | 485 | irregular | not circumscribed - indistinct | hypoechoic | shadowing | Suspicion of malignancy&Dysplasia | gate/gate/gate | 1.00 |
| case246.png | benign | 5 | 4.0 | 0.68 | 761 | irregular | not circumscribed - angular&indistinct | hypoechoic | shadowing | Suspicion of malignancy | gate/nomask/gate | 1.00 |
| case189.png | benign | 4b | 4.7 | 1.09 | 485 | oval | not circumscribed - indistinct | hypoechoic | no | Suspicion of malignancy&Dysplasia&Intraductal papilloma | gate/gate/gate | 1.00 |
| case075.png | malignant | 4c | 4.9 | 0.79 | 433 | irregular | not circumscribed - indistinct | hypoechoic | shadowing | Suspicion of malignancy&Dysplasia&Fibroadenoma&Intraductal p | gate/gate/gate | 1.00 |
| case023.png | benign | 4a | 4.9 | 1.37 | 545 | oval | circumscribed | hypoechoic | no | Cyst filled with thick fluid&Dysplasia&Fibroadenoma | gate/gate/gate | 1.00 |
| case065.png | benign | 2 | 5.0 | 1.39 | 545 | oval | circumscribed | hypoechoic | no | Cyst filled with thick fluid&Intramammary lymph node | gate/nomask/gate | 1.00 |
| case241.png | benign | 3 | 5.1 | 1.41 | 1024 | round | circumscribed | heterogeneous | no | Complex cyst / Non-simple cyst&Cyst filled with thick fluid& | gate/hit/gate | 0.33 |
| case017.png | benign | 3 | 5.2 | 1.47 | 930 | oval | circumscribed | complex cystic/solid | no | Dysplasia&Mammary duct ectasia | gate/gate/gate | 1.00 |
| case062.png | benign | 4b | 5.4 | 1.60 | 1024 | irregular | not circumscribed - indistinct | hypoechoic | shadowing | Suspicion of malignancy&Dysplasia | wrongloc/wrongloc/wrongloc | 0.67 |
| case171.png | benign | 4a | 5.4 | 1.46 | 485 | oval | circumscribed | hypoechoic | shadowing | Complex cyst / Non-simple cyst&Cyst filled with thick fluid& | gate/gate/gate | 0.67 |
| case054.png | benign | 3 | 5.7 | 2.08 | 930 | irregular | circumscribed | isoechoic | no | Dysplasia | gate/nomask/gate | 1.00 |
| case042.png | benign | 4b | 5.9 | 1.64 | 738 | round | circumscribed | isoechoic | no | Suspicion of malignancy&Intraductal papilloma | gate/hit/gate | 1.00 |
| case107.png | malignant | 4b | 6.0 | 1.20 | 433 | irregular | not circumscribed - indistinct | hypoechoic | no | Suspicion of malignancy&Dysplasia&Fibroadenoma&Intraductal p | gate/gate/gate | 1.00 |
| case034.png | malignant | 4c | 6.0 | 1.21 | 433 | irregular | not circumscribed - indistinct | hypoechoic | no | Suspicion of malignancy&Dysplasia&Fibroadenoma&Intraductal p | gate/gate/gate | 0.67 |
| case060.png | malignant | 4c | 6.2 | 2.16 | 545 | irregular | not circumscribed - indistinct | hypoechoic | shadowing | Suspicion of malignancy | gate/nomask/gate | 0.00 |
| case163.png | benign | 3 | 6.3 | 1.53 | 503 | oval | circumscribed | isoechoic | no | Suspicion of malignancy&Dysplasia&Fibroadenoma&Intraductal p | gate/gate/gate | 0.00 |
| case140.png | benign | 2 | 6.4 | 1.78 | 738 | oval | circumscribed | anechoic | no | Mammary duct ectasia | wrongloc/wrongloc/wrongloc | 0.33 |
| case136.png | benign | 2 | 6.4 | 1.65 | 738 | round | circumscribed | hyperechoic | no | Lipoma | gate/gate/gate | 1.00 |
| case070.png | benign | 4c | 6.5 | 2.74 | 623 | irregular | not circumscribed - indistinct | hypoechoic | shadowing | Suspicion of malignancy&Dysplasia | gate/hit/gate | 0.00 |
| case029.png | benign | 3 | 6.5 | 2.76 | 930 | irregular | circumscribed | isoechoic | no | Dysplasia&Fibroadenoma | gate/hit/gate | 1.00 |
| case059.png | benign | 4a | 6.8 | 2.57 | 545 | oval | circumscribed | hypoechoic | no | Dysplasia&Fibroadenoma&Duct filled with thick fluid&Intraduc | gate/gate/gate | 1.00 |
| case120.png | benign | 4b | 6.8 | 3.04 | 623 | irregular | not circumscribed - indistinct | hypoechoic | no | Suspicion of malignancy&Fibroadenoma | gate/nomask/gate | 0.00 |
| case130.png | malignant | 5 | 6.9 | 2.35 | 485 | irregular | not circumscribed - angular&indistinct | hypoechoic | no | Suspicion of malignancy | gate/gate/gate | 1.00 |
| case008.png | malignant | 5 | 7.3 | 2.69 | 794 | irregular | not circumscribed - spiculated&angular&indistinct | hypoechoic | shadowing | Suspicion of malignancy | gate/nomask/gate | 1.00 |
| case039.png | benign | 3 | 7.7 | 2.93 | 485 | oval | circumscribed | hypoechoic | no | Dysplasia&Fibroadenoma&Adenosis | gate/hit/gate | 0.33 |
| case250.png | malignant | 4c | 8.2 | 2.93 | 761 | irregular | not circumscribed - indistinct | heterogeneous | no | Suspicion of malignancy&Dysplasia | gate/nomask/gate | 1.00 |
| case071.png | benign | 4a | 9.6 | 5.15 | 545 | oval | circumscribed | heterogeneous | no | Hamartoma | gate/gate/hit | 0.00 |
| case187.png | benign | 4c | 9.7 | 3.85 | 761 | irregular | not circumscribed - indistinct | heterogeneous | no | Suspicion of malignancy&Dysplasia | gate/gate/gate | 1.00 |
| case167.png | benign | 2 | 10.0 | 3.46 | 887 | oval | circumscribed | hyperechoic | no | Lipoma&Hemangioma | gate/gate/gate | 0.00 |
| case022.png | benign | 4b | 10.1 | 2.93 | 831 | oval | circumscribed | hypoechoic | no | Dysplasia&Fibroadenoma&Intraductal papilloma | wrongloc/wrongloc/wrongloc | 1.00 |
| case252.png | benign | 4a | 11.3 | 7.20 | 545 | oval | circumscribed | hypoechoic | no | Fibroadenoma | gate/gate/gate | 1.00 |
| case231.png | benign | 4a | 11.5 | 6.51 | 485 | oval | circumscribed | isoechoic | no | Fibroadenoma | gate/gate/gate | 1.00 |
| case093.png | benign | 4a | 12.0 | 7.08 | 485 | oval | circumscribed | isoechoic | no | Fibroadenoma&Hamartoma | gate/gate/gate | 1.00 |
| case001.png | benign | 2 | 12.3 | 7.49 | 485 | irregular | not circumscribed - indistinct | heterogeneous | shadowing | Breast scar (surgery)&Breast scar (radiotherapy) | hit/nomask/gate | 0.00 |
| case115.png | benign | 5 | 12.7 | 5.70 | 656 | irregular | not circumscribed - indistinct | heterogeneous | no | Suspicion of malignancy&Dysplasia | gate/gate/gate | 0.33 |
| case232.png | benign | 4a | 13.9 | 8.38 | 761 | oval | circumscribed | isoechoic | enhancement | Dysplasia&Lipoma | gate/hit/gate | 0.00 |
| case228.png | malignant | 4b | 14.3 | 10.08 | 485 | irregular | not circumscribed - spiculated | heterogeneous | no | Suspicion of malignancy&Dysplasia&Duct filled with thick flu | wrongloc/wrongloc/wrongloc | 1.00 |
| case083.png | benign | 4a | 16.8 | 14.24 | 795 | irregular | circumscribed | heterogeneous | no | Dysplasia | gate/hit/gate | 0.00 |
| case194.png | benign | 2 | 17.3 | 16.95 | 797 | round | circumscribed | hyperechoic | enhancement | Implant rupture&Silicone implant | gate/gate/hit | 0.00 |
| case086.png | malignant | 5 | 22.3 | 16.51 | 433 | irregular | not circumscribed - indistinct | heterogeneous | shadowing | Suspicion of malignancy&Dysplasia&Fibroadenoma&Intraductal p | gate/gate/gate | 1.00 |
| case186.png | benign | 2 | 40.6 | 46.12 | 508 | irregular | circumscribed | hyperechoic | shadowing | Implant rupture&Silicone implant | gate/gate/gate | 0.33 |

## 6. Disagreement with the original method (consistent misses)

Both miss: 29 · only full-ViT misses: 15 · only LLM-LoRA misses: 5.

On lesions both methods hit in all seeds (n=187): Dice full-ViT 0.842 vs LLM-LoRA 0.829.
