# Stricter image-level localized F1

Hit = nonempty prediction with GT overlap and union-mask IoU >= the stated cutoff (256x256). TP = hit lesion image; FN = non-hit lesion image; FP = positive normal image plus nonempty non-hit lesion image. A localization failure with a nonempty prediction contributes both FP and FN. Each image contributes at most one TP. This is NOT instance-level detection F1; extra components in a hit image are penalized by union IoU, not counted as separate FP instances. All operating thresholds remain fixed from validation or the default raw/conf=0.25 rule; IoU cutoffs are evaluation criteria, not tuned operating thresholds.

Seed-pooled counts repeat the same test images; they are not independent patients. SD is across training seeds, not a CI.

| Set | Model | op | hit criterion | runs | TP | FP | FN | precision | recall | pooled F1 | F1 mean ± SD |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| busi:test | WeVisOne | raw | any_overlap | 3 | 195 | 17 | 3 | 0.920 | 0.985 | 0.951 | 0.951 ± 0.004 |
| busi:test | WeVisOne | raw | iou_0.25 | 3 | 185 | 27 | 13 | 0.873 | 0.934 | 0.902 | 0.902 ± 0.005 |
| busi:test | WeVisOne | raw | iou_0.50 | 3 | 177 | 35 | 21 | 0.835 | 0.894 | 0.863 | 0.863 ± 0.012 |
| busi:test | WeVisOne | gate | any_overlap | 3 | 195 | 15 | 3 | 0.929 | 0.985 | 0.956 | 0.956 ± 0.007 |
| busi:test | WeVisOne | gate | iou_0.25 | 3 | 185 | 25 | 13 | 0.881 | 0.934 | 0.907 | 0.907 ± 0.011 |
| busi:test | WeVisOne | gate | iou_0.50 | 3 | 177 | 33 | 21 | 0.843 | 0.894 | 0.868 | 0.868 ± 0.013 |
| busi:test | WeVisOne strong-aug | raw | any_overlap | 3 | 193 | 13 | 5 | 0.937 | 0.975 | 0.955 | 0.955 ± 0.000 |
| busi:test | WeVisOne strong-aug | raw | iou_0.25 | 3 | 183 | 23 | 15 | 0.888 | 0.924 | 0.906 | 0.906 ± 0.009 |
| busi:test | WeVisOne strong-aug | raw | iou_0.50 | 3 | 167 | 39 | 31 | 0.811 | 0.843 | 0.827 | 0.827 ± 0.018 |
| busi:test | WeVisOne strong-aug | gate | any_overlap | 3 | 192 | 11 | 6 | 0.946 | 0.970 | 0.958 | 0.958 ± 0.005 |
| busi:test | WeVisOne strong-aug | gate | iou_0.25 | 3 | 182 | 21 | 16 | 0.897 | 0.919 | 0.908 | 0.908 ± 0.013 |
| busi:test | WeVisOne strong-aug | gate | iou_0.50 | 3 | 167 | 36 | 31 | 0.823 | 0.843 | 0.833 | 0.833 ± 0.013 |
| busi:test | YOLO26m-seg | 0.25 | any_overlap | 3 | 184 | 7 | 14 | 0.963 | 0.929 | 0.946 | 0.946 ± 0.011 |
| busi:test | YOLO26m-seg | 0.25 | iou_0.25 | 3 | 172 | 19 | 26 | 0.901 | 0.869 | 0.884 | 0.885 ± 0.035 |
| busi:test | YOLO26m-seg | 0.25 | iou_0.50 | 3 | 163 | 28 | 35 | 0.853 | 0.823 | 0.838 | 0.839 ± 0.052 |
| busi:test | YOLO26m-seg | cal | any_overlap | 3 | 158 | 1 | 40 | 0.994 | 0.798 | 0.885 | 0.884 ± 0.030 |
| busi:test | YOLO26m-seg | cal | iou_0.25 | 3 | 153 | 6 | 45 | 0.962 | 0.773 | 0.857 | 0.856 ± 0.038 |
| busi:test | YOLO26m-seg | cal | iou_0.50 | 3 | 151 | 8 | 47 | 0.950 | 0.763 | 0.846 | 0.845 ± 0.024 |
| busi:test | nnU-Net v2 2d | raw | any_overlap | 1 | 66 | 3 | 0 | 0.957 | 1.000 | 0.978 | 0.978 ± 0.000 |
| busi:test | nnU-Net v2 2d | raw | iou_0.25 | 1 | 62 | 7 | 4 | 0.899 | 0.939 | 0.919 | 0.919 ± 0.000 |
| busi:test | nnU-Net v2 2d | raw | iou_0.50 | 1 | 57 | 12 | 9 | 0.826 | 0.864 | 0.844 | 0.844 ± 0.000 |
| busi:test | nnU-Net v2 2d | gate | any_overlap | 1 | 47 | 0 | 19 | 1.000 | 0.712 | 0.832 | 0.832 ± 0.000 |
| busi:test | nnU-Net v2 2d | gate | iou_0.25 | 1 | 47 | 0 | 19 | 1.000 | 0.712 | 0.832 | 0.832 ± 0.000 |
| busi:test | nnU-Net v2 2d | gate | iou_0.50 | 1 | 47 | 0 | 19 | 1.000 | 0.712 | 0.832 | 0.832 ± 0.000 |
| uclm:test | WeVisOne | raw | any_overlap | 3 | 387 | 108 | 42 | 0.782 | 0.902 | 0.838 | 0.838 ± 0.031 |
| uclm:test | WeVisOne | raw | iou_0.25 | 3 | 338 | 157 | 91 | 0.683 | 0.788 | 0.732 | 0.732 ± 0.021 |
| uclm:test | WeVisOne | raw | iou_0.50 | 3 | 322 | 173 | 107 | 0.651 | 0.751 | 0.697 | 0.697 ± 0.006 |
| uclm:test | WeVisOne | gate | any_overlap | 3 | 382 | 94 | 47 | 0.803 | 0.890 | 0.844 | 0.845 ± 0.031 |
| uclm:test | WeVisOne | gate | iou_0.25 | 3 | 334 | 142 | 95 | 0.702 | 0.779 | 0.738 | 0.739 ± 0.023 |
| uclm:test | WeVisOne | gate | iou_0.50 | 3 | 318 | 158 | 111 | 0.668 | 0.741 | 0.703 | 0.703 ± 0.008 |
| uclm:test | WeVisOne strong-aug | raw | any_overlap | 3 | 383 | 62 | 46 | 0.861 | 0.893 | 0.876 | 0.877 ± 0.011 |
| uclm:test | WeVisOne strong-aug | raw | iou_0.25 | 3 | 328 | 117 | 101 | 0.737 | 0.765 | 0.751 | 0.751 ± 0.013 |
| uclm:test | WeVisOne strong-aug | raw | iou_0.50 | 3 | 306 | 139 | 123 | 0.688 | 0.713 | 0.700 | 0.700 ± 0.013 |
| uclm:test | WeVisOne strong-aug | gate | any_overlap | 3 | 378 | 54 | 51 | 0.875 | 0.881 | 0.878 | 0.878 ± 0.009 |
| uclm:test | WeVisOne strong-aug | gate | iou_0.25 | 3 | 327 | 105 | 102 | 0.757 | 0.762 | 0.760 | 0.760 ± 0.020 |
| uclm:test | WeVisOne strong-aug | gate | iou_0.50 | 3 | 306 | 126 | 123 | 0.708 | 0.713 | 0.711 | 0.711 ± 0.018 |
| uclm:test | YOLO26m-seg | 0.25 | any_overlap | 3 | 301 | 279 | 128 | 0.519 | 0.702 | 0.597 | 0.608 ± 0.111 |
| uclm:test | YOLO26m-seg | 0.25 | iou_0.25 | 3 | 286 | 294 | 143 | 0.493 | 0.667 | 0.567 | 0.577 ± 0.097 |
| uclm:test | YOLO26m-seg | 0.25 | iou_0.50 | 3 | 260 | 320 | 169 | 0.448 | 0.606 | 0.515 | 0.527 ± 0.117 |
| uclm:test | YOLO26m-seg | cal | any_overlap | 3 | 243 | 73 | 186 | 0.769 | 0.566 | 0.652 | 0.653 ± 0.024 |
| uclm:test | YOLO26m-seg | cal | iou_0.25 | 3 | 239 | 77 | 190 | 0.756 | 0.557 | 0.642 | 0.642 ± 0.028 |
| uclm:test | YOLO26m-seg | cal | iou_0.50 | 3 | 233 | 83 | 196 | 0.737 | 0.543 | 0.626 | 0.626 ± 0.027 |
| uclm:test | nnU-Net v2 2d | raw | any_overlap | 1 | 112 | 143 | 31 | 0.439 | 0.783 | 0.563 | 0.563 ± 0.000 |
| uclm:test | nnU-Net v2 2d | raw | iou_0.25 | 1 | 99 | 156 | 44 | 0.388 | 0.692 | 0.497 | 0.497 ± 0.000 |
| uclm:test | nnU-Net v2 2d | raw | iou_0.50 | 1 | 80 | 175 | 63 | 0.314 | 0.559 | 0.402 | 0.402 ± 0.000 |
| uclm:test | nnU-Net v2 2d | gate | any_overlap | 1 | 41 | 8 | 102 | 0.837 | 0.287 | 0.427 | 0.427 ± 0.000 |
| uclm:test | nnU-Net v2 2d | gate | iou_0.25 | 1 | 40 | 9 | 103 | 0.816 | 0.280 | 0.417 | 0.417 ± 0.000 |
| uclm:test | nnU-Net v2 2d | gate | iou_0.50 | 1 | 38 | 11 | 105 | 0.776 | 0.266 | 0.396 | 0.396 ± 0.000 |
| uclm:all | WeVisOne | raw | any_overlap | 3 | 650 | 335 | 130 | 0.660 | 0.833 | 0.737 | 0.738 ± 0.033 |
| uclm:all | WeVisOne | raw | iou_0.25 | 3 | 540 | 445 | 240 | 0.548 | 0.692 | 0.612 | 0.612 ± 0.018 |
| uclm:all | WeVisOne | raw | iou_0.50 | 3 | 488 | 497 | 292 | 0.495 | 0.626 | 0.553 | 0.553 ± 0.009 |
| uclm:all | WeVisOne | gate | any_overlap | 3 | 644 | 297 | 136 | 0.684 | 0.826 | 0.748 | 0.751 ± 0.046 |
| uclm:all | WeVisOne | gate | iou_0.25 | 3 | 536 | 405 | 244 | 0.570 | 0.687 | 0.623 | 0.625 ± 0.031 |
| uclm:all | WeVisOne | gate | iou_0.50 | 3 | 484 | 457 | 296 | 0.514 | 0.621 | 0.562 | 0.564 ± 0.021 |
| uclm:all | WeVisOne strong-aug | raw | any_overlap | 3 | 637 | 215 | 143 | 0.748 | 0.817 | 0.781 | 0.781 ± 0.015 |
| uclm:all | WeVisOne strong-aug | raw | iou_0.25 | 3 | 509 | 343 | 271 | 0.597 | 0.653 | 0.624 | 0.624 ± 0.019 |
| uclm:all | WeVisOne strong-aug | raw | iou_0.50 | 3 | 458 | 394 | 322 | 0.538 | 0.587 | 0.561 | 0.561 ± 0.011 |
| uclm:all | WeVisOne strong-aug | gate | any_overlap | 3 | 623 | 195 | 157 | 0.762 | 0.799 | 0.780 | 0.780 ± 0.015 |
| uclm:all | WeVisOne strong-aug | gate | iou_0.25 | 3 | 506 | 312 | 274 | 0.619 | 0.649 | 0.633 | 0.634 ± 0.027 |
| uclm:all | WeVisOne strong-aug | gate | iou_0.50 | 3 | 456 | 362 | 324 | 0.557 | 0.585 | 0.571 | 0.571 ± 0.019 |
| uclm:all | YOLO26m-seg | 0.25 | any_overlap | 3 | 464 | 658 | 316 | 0.414 | 0.595 | 0.488 | 0.499 ± 0.107 |
| uclm:all | YOLO26m-seg | 0.25 | iou_0.25 | 3 | 435 | 687 | 345 | 0.388 | 0.558 | 0.457 | 0.467 ± 0.091 |
| uclm:all | YOLO26m-seg | 0.25 | iou_0.50 | 3 | 379 | 743 | 401 | 0.338 | 0.486 | 0.399 | 0.411 ± 0.108 |
| uclm:all | YOLO26m-seg | cal | any_overlap | 3 | 341 | 184 | 439 | 0.650 | 0.437 | 0.523 | 0.523 ± 0.013 |
| uclm:all | YOLO26m-seg | cal | iou_0.25 | 3 | 331 | 194 | 449 | 0.630 | 0.424 | 0.507 | 0.508 ± 0.011 |
| uclm:all | YOLO26m-seg | cal | iou_0.50 | 3 | 317 | 208 | 463 | 0.604 | 0.406 | 0.486 | 0.487 ± 0.022 |
| uclm:all | nnU-Net v2 2d | raw | any_overlap | 1 | 192 | 333 | 68 | 0.366 | 0.738 | 0.489 | 0.489 ± 0.000 |
| uclm:all | nnU-Net v2 2d | raw | iou_0.25 | 1 | 154 | 371 | 106 | 0.293 | 0.592 | 0.392 | 0.392 ± 0.000 |
| uclm:all | nnU-Net v2 2d | raw | iou_0.50 | 1 | 113 | 412 | 147 | 0.215 | 0.435 | 0.288 | 0.288 ± 0.000 |
| uclm:all | nnU-Net v2 2d | gate | any_overlap | 1 | 48 | 20 | 212 | 0.706 | 0.185 | 0.293 | 0.293 ± 0.000 |
| uclm:all | nnU-Net v2 2d | gate | iou_0.25 | 1 | 46 | 22 | 214 | 0.676 | 0.177 | 0.280 | 0.280 ± 0.000 |
| uclm:all | nnU-Net v2 2d | gate | iou_0.50 | 1 | 42 | 26 | 218 | 0.618 | 0.162 | 0.256 | 0.256 ± 0.000 |
| busuc:all | WeVisOne | raw | any_overlap | 3 | 2411 | 20 | 22 | 0.992 | 0.991 | 0.991 | 0.991 ± 0.001 |
| busuc:all | WeVisOne | raw | iou_0.25 | 3 | 2363 | 68 | 70 | 0.972 | 0.971 | 0.972 | 0.972 ± 0.001 |
| busuc:all | WeVisOne | raw | iou_0.50 | 3 | 2293 | 138 | 140 | 0.943 | 0.942 | 0.943 | 0.943 ± 0.002 |
| busuc:all | WeVisOne | gate | any_overlap | 3 | 2408 | 20 | 25 | 0.992 | 0.990 | 0.991 | 0.991 ± 0.002 |
| busuc:all | WeVisOne | gate | iou_0.25 | 3 | 2362 | 66 | 71 | 0.973 | 0.971 | 0.972 | 0.972 ± 0.001 |
| busuc:all | WeVisOne | gate | iou_0.50 | 3 | 2293 | 135 | 140 | 0.944 | 0.942 | 0.943 | 0.943 ± 0.002 |
| busuc:all | WeVisOne strong-aug | raw | any_overlap | 3 | 2412 | 21 | 21 | 0.991 | 0.991 | 0.991 | 0.991 ± 0.001 |
| busuc:all | WeVisOne strong-aug | raw | iou_0.25 | 3 | 2205 | 228 | 228 | 0.906 | 0.906 | 0.906 | 0.906 ± 0.019 |
| busuc:all | WeVisOne strong-aug | raw | iou_0.50 | 3 | 2089 | 344 | 344 | 0.859 | 0.859 | 0.859 | 0.859 ± 0.023 |
| busuc:all | WeVisOne strong-aug | gate | any_overlap | 3 | 2410 | 21 | 23 | 0.991 | 0.991 | 0.991 | 0.991 ± 0.001 |
| busuc:all | WeVisOne strong-aug | gate | iou_0.25 | 3 | 2205 | 226 | 228 | 0.907 | 0.906 | 0.907 | 0.907 ± 0.019 |
| busuc:all | WeVisOne strong-aug | gate | iou_0.50 | 3 | 2089 | 342 | 344 | 0.859 | 0.859 | 0.859 | 0.859 ± 0.023 |
| busuc:all | YOLO26m-seg | 0.25 | any_overlap | 3 | 1960 | 46 | 473 | 0.977 | 0.806 | 0.883 | 0.881 ± 0.056 |
| busuc:all | YOLO26m-seg | 0.25 | iou_0.25 | 3 | 1795 | 211 | 638 | 0.895 | 0.738 | 0.809 | 0.809 ± 0.014 |
| busuc:all | YOLO26m-seg | 0.25 | iou_0.50 | 3 | 1557 | 449 | 876 | 0.776 | 0.640 | 0.702 | 0.703 ± 0.057 |
| busuc:all | YOLO26m-seg | cal | any_overlap | 3 | 1399 | 22 | 1034 | 0.985 | 0.575 | 0.726 | 0.725 ± 0.049 |
| busuc:all | YOLO26m-seg | cal | iou_0.25 | 3 | 1381 | 40 | 1052 | 0.972 | 0.568 | 0.717 | 0.715 ± 0.046 |
| busuc:all | YOLO26m-seg | cal | iou_0.50 | 3 | 1347 | 74 | 1086 | 0.948 | 0.554 | 0.699 | 0.698 ± 0.042 |
| busuc:all | nnU-Net v2 2d | raw | any_overlap | 1 | 804 | 6 | 7 | 0.993 | 0.991 | 0.992 | 0.992 ± 0.000 |
| busuc:all | nnU-Net v2 2d | raw | iou_0.25 | 1 | 773 | 37 | 38 | 0.954 | 0.953 | 0.954 | 0.954 ± 0.000 |
| busuc:all | nnU-Net v2 2d | raw | iou_0.50 | 1 | 707 | 103 | 104 | 0.873 | 0.872 | 0.872 | 0.872 ± 0.000 |
| busuc:all | nnU-Net v2 2d | gate | any_overlap | 1 | 538 | 0 | 273 | 1.000 | 0.663 | 0.798 | 0.798 ± 0.000 |
| busuc:all | nnU-Net v2 2d | gate | iou_0.25 | 1 | 534 | 4 | 277 | 0.993 | 0.658 | 0.792 | 0.792 ± 0.000 |
| busuc:all | nnU-Net v2 2d | gate | iou_0.50 | 1 | 501 | 37 | 310 | 0.931 | 0.618 | 0.743 | 0.743 ± 0.000 |
| busbra:test | WeVisOne | raw | any_overlap | 3 | 1073 | 47 | 49 | 0.958 | 0.956 | 0.957 | 0.957 ± 0.005 |
| busbra:test | WeVisOne | raw | iou_0.25 | 3 | 977 | 143 | 145 | 0.872 | 0.871 | 0.872 | 0.872 ± 0.017 |
| busbra:test | WeVisOne | raw | iou_0.50 | 3 | 830 | 290 | 292 | 0.741 | 0.740 | 0.740 | 0.740 ± 0.014 |
| busbra:test | WeVisOne | gate | any_overlap | 3 | 1070 | 47 | 52 | 0.958 | 0.954 | 0.956 | 0.956 ± 0.006 |
| busbra:test | WeVisOne | gate | iou_0.25 | 3 | 976 | 141 | 146 | 0.874 | 0.870 | 0.872 | 0.872 ± 0.017 |
| busbra:test | WeVisOne | gate | iou_0.50 | 3 | 829 | 288 | 293 | 0.742 | 0.739 | 0.741 | 0.741 ± 0.014 |
| busbra:test | WeVisOne strong-aug | raw | any_overlap | 3 | 1081 | 39 | 41 | 0.965 | 0.963 | 0.964 | 0.964 ± 0.006 |
| busbra:test | WeVisOne strong-aug | raw | iou_0.25 | 3 | 834 | 286 | 288 | 0.745 | 0.743 | 0.744 | 0.744 ± 0.027 |
| busbra:test | WeVisOne strong-aug | raw | iou_0.50 | 3 | 680 | 440 | 442 | 0.607 | 0.606 | 0.607 | 0.607 ± 0.020 |
| busbra:test | WeVisOne strong-aug | gate | any_overlap | 3 | 1073 | 39 | 49 | 0.965 | 0.956 | 0.961 | 0.961 ± 0.002 |
| busbra:test | WeVisOne strong-aug | gate | iou_0.25 | 3 | 833 | 279 | 289 | 0.749 | 0.742 | 0.746 | 0.746 ± 0.027 |
| busbra:test | WeVisOne strong-aug | gate | iou_0.50 | 3 | 680 | 432 | 442 | 0.612 | 0.606 | 0.609 | 0.609 ± 0.020 |
| busbra:test | YOLO26m-seg | 0.25 | any_overlap | 3 | 575 | 43 | 547 | 0.930 | 0.512 | 0.661 | 0.650 ± 0.137 |
| busbra:test | YOLO26m-seg | 0.25 | iou_0.25 | 3 | 522 | 96 | 600 | 0.845 | 0.465 | 0.600 | 0.595 ± 0.064 |
| busbra:test | YOLO26m-seg | 0.25 | iou_0.50 | 3 | 460 | 158 | 662 | 0.744 | 0.410 | 0.529 | 0.528 ± 0.021 |
| busbra:test | YOLO26m-seg | cal | any_overlap | 3 | 379 | 11 | 743 | 0.972 | 0.338 | 0.501 | 0.500 ± 0.051 |
| busbra:test | YOLO26m-seg | cal | iou_0.25 | 3 | 378 | 12 | 744 | 0.969 | 0.337 | 0.500 | 0.499 ± 0.049 |
| busbra:test | YOLO26m-seg | cal | iou_0.50 | 3 | 351 | 39 | 771 | 0.900 | 0.313 | 0.464 | 0.464 ± 0.031 |
| busbra:test | nnU-Net v2 2d | raw | any_overlap | 1 | 330 | 7 | 44 | 0.979 | 0.882 | 0.928 | 0.928 ± 0.000 |
| busbra:test | nnU-Net v2 2d | raw | iou_0.25 | 1 | 301 | 36 | 73 | 0.893 | 0.805 | 0.847 | 0.847 ± 0.000 |
| busbra:test | nnU-Net v2 2d | raw | iou_0.50 | 1 | 259 | 78 | 115 | 0.769 | 0.693 | 0.729 | 0.729 ± 0.000 |
| busbra:test | nnU-Net v2 2d | gate | any_overlap | 1 | 126 | 1 | 248 | 0.992 | 0.337 | 0.503 | 0.503 ± 0.000 |
| busbra:test | nnU-Net v2 2d | gate | iou_0.25 | 1 | 125 | 2 | 249 | 0.984 | 0.334 | 0.499 | 0.499 ± 0.000 |
| busbra:test | nnU-Net v2 2d | gate | iou_0.50 | 1 | 114 | 13 | 260 | 0.898 | 0.305 | 0.455 | 0.455 ± 0.000 |
| breast:all | WeVisOne | raw | any_overlap | 3 | 656 | 26 | 100 | 0.962 | 0.868 | 0.912 | 0.912 ± 0.007 |
| breast:all | WeVisOne | raw | iou_0.25 | 3 | 598 | 84 | 158 | 0.877 | 0.791 | 0.832 | 0.832 ± 0.012 |
| breast:all | WeVisOne | raw | iou_0.50 | 3 | 556 | 126 | 200 | 0.815 | 0.735 | 0.773 | 0.773 ± 0.013 |
| breast:all | WeVisOne | gate | any_overlap | 3 | 647 | 24 | 109 | 0.964 | 0.856 | 0.907 | 0.907 ± 0.009 |
| breast:all | WeVisOne | gate | iou_0.25 | 3 | 593 | 78 | 163 | 0.884 | 0.784 | 0.831 | 0.831 ± 0.010 |
| breast:all | WeVisOne | gate | iou_0.50 | 3 | 551 | 120 | 205 | 0.821 | 0.729 | 0.772 | 0.772 ± 0.013 |
| breast:all | WeVisOne strong-aug | raw | any_overlap | 3 | 654 | 15 | 102 | 0.978 | 0.865 | 0.918 | 0.918 ± 0.014 |
| breast:all | WeVisOne strong-aug | raw | iou_0.25 | 3 | 546 | 123 | 210 | 0.816 | 0.722 | 0.766 | 0.766 ± 0.029 |
| breast:all | WeVisOne strong-aug | raw | iou_0.50 | 3 | 507 | 162 | 249 | 0.758 | 0.671 | 0.712 | 0.711 ± 0.029 |
| breast:all | WeVisOne strong-aug | gate | any_overlap | 3 | 638 | 15 | 118 | 0.977 | 0.844 | 0.906 | 0.905 ± 0.013 |
| breast:all | WeVisOne strong-aug | gate | iou_0.25 | 3 | 539 | 114 | 217 | 0.825 | 0.713 | 0.765 | 0.765 ± 0.028 |
| breast:all | WeVisOne strong-aug | gate | iou_0.50 | 3 | 500 | 153 | 256 | 0.766 | 0.661 | 0.710 | 0.710 ± 0.028 |
| breast:all | YOLO26m-seg | 0.25 | any_overlap | 3 | 414 | 27 | 342 | 0.939 | 0.548 | 0.692 | 0.687 ± 0.081 |
| breast:all | YOLO26m-seg | 0.25 | iou_0.25 | 3 | 387 | 54 | 369 | 0.878 | 0.512 | 0.647 | 0.643 ± 0.053 |
| breast:all | YOLO26m-seg | 0.25 | iou_0.50 | 3 | 348 | 93 | 408 | 0.789 | 0.460 | 0.581 | 0.579 ± 0.041 |
| breast:all | YOLO26m-seg | cal | any_overlap | 3 | 291 | 7 | 465 | 0.977 | 0.385 | 0.552 | 0.552 ± 0.015 |
| breast:all | YOLO26m-seg | cal | iou_0.25 | 3 | 284 | 14 | 472 | 0.953 | 0.376 | 0.539 | 0.539 ± 0.012 |
| breast:all | YOLO26m-seg | cal | iou_0.50 | 3 | 269 | 29 | 487 | 0.903 | 0.356 | 0.510 | 0.510 ± 0.013 |
| breast:all | nnU-Net v2 2d | raw | any_overlap | 1 | 180 | 12 | 72 | 0.938 | 0.714 | 0.811 | 0.811 ± 0.000 |
| breast:all | nnU-Net v2 2d | raw | iou_0.25 | 1 | 164 | 28 | 88 | 0.854 | 0.651 | 0.739 | 0.739 ± 0.000 |
| breast:all | nnU-Net v2 2d | raw | iou_0.50 | 1 | 143 | 49 | 109 | 0.745 | 0.567 | 0.644 | 0.644 ± 0.000 |
| breast:all | nnU-Net v2 2d | gate | any_overlap | 1 | 66 | 1 | 186 | 0.985 | 0.262 | 0.414 | 0.414 ± 0.000 |
| breast:all | nnU-Net v2 2d | gate | iou_0.25 | 1 | 63 | 4 | 189 | 0.940 | 0.250 | 0.395 | 0.395 ± 0.000 |
| breast:all | nnU-Net v2 2d | gate | iou_0.50 | 1 | 59 | 8 | 193 | 0.881 | 0.234 | 0.370 | 0.370 ± 0.000 |