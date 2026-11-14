| Set | prompt | FPR % (FP/N) | lesion Dice | lesion empty % | miss % | AUROC | gate FPR % | gate lesion Dice | gate miss % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| busi:val | original | 15.4 (2/13) | 0.826 | 2.8 | 4.2 | 0.989 | 0.0 | 0.818 | 5.6 |
| busi:val | +None hint | 7.7 (1/13) | 0.809 | 6.9 | 8.3 | 0.987 | 0.0 | 0.809 | 8.3 |
| busi:val | flips | normal fixed 1, normal broken 0, lesion lost 3, lesion gained 0 ||||||||
| busi:test | original | 50.0 (5/10) | 0.833 | 0.0 | 1.5 | 0.963 | 30.0 | 0.833 | 1.5 |
| busi:test | +None hint | 50.0 (5/10) | 0.828 | 0.0 | 1.5 | 0.968 | 30.0 | 0.826 | 3.0 |
| busi:test | flips | normal fixed 0, normal broken 0, lesion lost 0, lesion gained 0 ||||||||
| uclm:test | original | 12.2 (23/188) | 0.693 | 10.5 | 11.2 | 0.957 | 8.0 | 0.674 | 14.0 |
| uclm:test | +None hint | 10.1 (19/188) | 0.715 | 7.7 | 9.1 | 0.964 | 6.9 | 0.680 | 13.3 |
| uclm:test | flips | normal fixed 7, normal broken 3, lesion lost 2, lesion gained 5 ||||||||
| uclm:all | original | 15.5 (65/419) | 0.592 | 11.9 | 17.7 | 0.940 | 9.3 | 0.581 | 19.6 |
| uclm:all | +None hint | 12.6 (53/419) | 0.615 | 8.8 | 15.8 | 0.947 | 8.1 | 0.585 | 19.2 |
| uclm:all | flips | normal fixed 22, normal broken 10, lesion lost 3, lesion gained 8 ||||||||