# Zero-shot vs fine-tuned WeVisOne, original vs '+None' prompt

k/N summed over seeds. FPR = normal images with any predicted pixel; zero-overlap = lesion images whose prediction does not touch the GT; gate = presence threshold from busi:val normals. The '+None' prompt was proposed after looking at UCLM errors, so its UCLM numbers are exploratory.

| Set | Model | op | FPR (k/N) | lesion Dice (mean ± SD over seeds) | zero-overlap (k/N) |
|---|---|---|---:|---:|---:|
| busi:test | zero-shot (n=1) | raw | 100.0% (10/10) | 0.700 ± 0.000 | 3.0% (2/66) |
| busi:test | zero-shot (n=1) | gate | 0.0% (0/10) | 0.061 ± 0.000 | 89.4% (59/66) |
| busi:test | fine-tuned, original prompt (n=3) | raw | 46.7% (14/30) | 0.829 ± 0.010 | 1.5% (3/198) |
| busi:test | fine-tuned, original prompt (n=3) | gate | 40.0% (12/30) | 0.829 ± 0.010 | 1.5% (3/198) |
| busi:test | fine-tuned, +None prompt (n=3) | raw | 40.0% (12/30) | 0.825 ± 0.002 | 1.5% (3/198) |
| busi:test | fine-tuned, +None prompt (n=3) | gate | 33.3% (10/30) | 0.824 ± 0.002 | 2.5% (5/198) |
| uclm:test | zero-shot (n=1) | raw | 100.0% (188/188) | 0.648 ± 0.000 | 0.7% (1/143) |
| uclm:test | zero-shot (n=1) | gate | 43.6% (82/188) | 0.516 ± 0.000 | 28.0% (40/143) |
| uclm:test | fine-tuned, original prompt (n=3) | raw | 17.7% (100/564) | 0.716 ± 0.022 | 9.8% (42/429) |
| uclm:test | fine-tuned, original prompt (n=3) | gate | 15.4% (87/564) | 0.708 ± 0.032 | 11.0% (47/429) |
| uclm:test | fine-tuned, +None prompt (n=3) | raw | 23.2% (131/564) | 0.729 ± 0.024 | 8.4% (36/429) |
| uclm:test | fine-tuned, +None prompt (n=3) | gate | 15.8% (89/564) | 0.709 ± 0.026 | 10.7% (46/429) |
| uclm:all | zero-shot (n=1) | raw | 100.0% (419/419) | 0.554 ± 0.000 | 5.0% (13/260) |
| uclm:all | zero-shot (n=1) | gate | 42.0% (176/419) | 0.404 ± 0.000 | 38.8% (101/260) |
| uclm:all | fine-tuned, original prompt (n=3) | raw | 21.7% (273/1257) | 0.611 ± 0.017 | 16.7% (130/780) |
| uclm:all | fine-tuned, original prompt (n=3) | gate | 18.9% (237/1257) | 0.607 ± 0.023 | 17.4% (136/780) |
| uclm:all | fine-tuned, +None prompt (n=3) | raw | 25.9% (325/1257) | 0.623 ± 0.015 | 15.6% (122/780) |
| uclm:all | fine-tuned, +None prompt (n=3) | gate | 18.8% (236/1257) | 0.607 ± 0.020 | 17.4% (136/780) |
| busuc:all | zero-shot (n=1) | raw | – | 0.880 ± 0.000 | 0.2% (2/811) |
| busuc:all | zero-shot (n=1) | gate | – | 0.000 ± 0.000 | 100.0% (811/811) |
| busuc:all | fine-tuned, original prompt (n=3) | raw | – | 0.880 ± 0.001 | 0.9% (22/2433) |
| busuc:all | fine-tuned, original prompt (n=3) | gate | – | 0.879 ± 0.001 | 1.0% (25/2433) |
| busuc:all | fine-tuned, +None prompt (n=3) | raw | – | 0.876 ± 0.007 | 1.5% (37/2433) |
| busuc:all | fine-tuned, +None prompt (n=3) | gate | – | 0.876 ± 0.007 | 1.5% (37/2433) |
| busbra:test | zero-shot (n=1) | raw | – | 0.545 ± 0.000 | 2.9% (11/374) |
| busbra:test | zero-shot (n=1) | gate | – | 0.059 ± 0.000 | 92.2% (345/374) |
| busbra:test | fine-tuned, original prompt (n=3) | raw | – | 0.733 ± 0.009 | 4.4% (49/1122) |
| busbra:test | fine-tuned, original prompt (n=3) | gate | – | 0.732 ± 0.008 | 4.6% (52/1122) |
| busbra:test | fine-tuned, +None prompt (n=3) | raw | – | 0.712 ± 0.011 | 7.0% (79/1122) |
| busbra:test | fine-tuned, +None prompt (n=3) | gate | – | 0.712 ± 0.011 | 7.0% (79/1122) |
| breast:all | zero-shot (n=1) | raw | 100.0% (4/4) | 0.618 ± 0.000 | 2.4% (6/252) |
| breast:all | zero-shot (n=1) | gate | 0.0% (0/4) | 0.084 ± 0.000 | 86.5% (218/252) |
| breast:all | fine-tuned, original prompt (n=3) | raw | 41.7% (5/12) | 0.701 ± 0.011 | 13.2% (100/756) |
| breast:all | fine-tuned, original prompt (n=3) | gate | 33.3% (4/12) | 0.694 ± 0.015 | 14.4% (109/756) |
| breast:all | fine-tuned, +None prompt (n=3) | raw | 41.7% (5/12) | 0.676 ± 0.029 | 17.3% (131/756) |
| breast:all | fine-tuned, +None prompt (n=3) | gate | 41.7% (5/12) | 0.672 ± 0.026 | 18.0% (136/756) |