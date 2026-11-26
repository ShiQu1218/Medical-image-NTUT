# Pre-registered: full-ViT vs YOLO26m, trained on BUSI + BUS-BRA + UCLM train

Pre-registration verified unchanged (sha256 ad6bedf93092). 3 seeds per arm; per-source gate rules.

## uclm:test

| Arm | rule | lesion Dice | per seed | miss % | normal FPR % |
|---|---|---:|---|---:|---:|
| full-ViT | q90 | 0.785 ± 0.025 | 0.811 / 0.762 / 0.781 | 12.6 | 5.9 |
| full-ViT | max | 0.771 ± 0.036 | 0.811 / 0.762 / 0.740 | 14.5 | 4.6 |
| full-ViT | raw | 0.785 ± 0.025 | 0.811 / 0.762 / 0.781 | 12.6 | 5.9 |
| LLM-LoRA | q90 | 0.684 ± 0.045 | 0.734 / 0.675 / 0.645 | 21.7 | 4.4 |
| LLM-LoRA | max | 0.641 ± 0.039 | 0.649 / 0.675 / 0.599 | 27.5 | 2.7 |
| LLM-LoRA | raw | 0.684 ± 0.045 | 0.734 / 0.675 / 0.645 | 21.7 | 4.4 |
| YOLO26m | q90 | 0.771 ± 0.026 | 0.747 / 0.769 / 0.798 | 9.3 | 15.1 |
| YOLO26m | max | 0.734 ± 0.021 | 0.721 / 0.758 / 0.722 | 17.5 | 4.6 |
| YOLO26m | raw | 0.768 ± 0.026 | 0.739 / 0.778 / 0.786 | 11.7 | 8.9 |
| full-ViT balanced (exploratory) | q90 | 0.828 ± 0.021 | 0.853 / 0.817 / 0.816 | 8.6 | 4.3 |
| full-ViT balanced (exploratory) | max | 0.720 ± 0.061 | 0.695 / 0.790 / 0.676 | 21.7 | 1.8 |
| full-ViT balanced (exploratory) | raw | 0.828 ± 0.021 | 0.853 / 0.817 / 0.816 | 8.6 | 4.3 |

| Comparison (patient-cluster bootstrap) | rule | dDice | 95% CI | p | dFPR | verdict |
|---|---|---:|---|---:|---:|---|
| full-ViT − YOLO26m **(PRIMARY)** | q90 | +0.013 | [-0.027, +0.051] | 0.5476 | -9.2 | INCONCLUSIVE |
| full-ViT − YOLO26m | max | +0.037 | [+0.002, +0.063] | 0.0412 | -0.0 | WIN |
| full-ViT − YOLO26m | raw | +0.017 | [-0.019, +0.050] | 0.3704 | -3.0 | INCONCLUSIVE |
| full-ViT − LLM-LoRA | q90 | +0.100 | [+0.041, +0.151] | 0.0004 | +1.4 |  |
| full-ViT − LLM-LoRA | max | +0.130 | [+0.066, +0.187] | 0.0002 | +2.0 |  |
| full-ViT − LLM-LoRA | raw | +0.100 | [+0.041, +0.151] | 0.0004 | +1.4 |  |
| LLM-LoRA − YOLO26m | q90 | -0.087 | [-0.142, -0.036] | 0.0008 | -10.6 |  |
| LLM-LoRA − YOLO26m | max | -0.093 | [-0.145, -0.033] | 0.0028 | -2.0 |  |
| LLM-LoRA − YOLO26m | raw | -0.083 | [-0.146, -0.023] | 0.0064 | -4.4 |  |
| full-ViT balanced (exploratory) − YOLO26m | q90 | +0.057 | [+0.015, +0.089] | 0.0092 | -10.8 |  |
| full-ViT balanced (exploratory) − YOLO26m | max | -0.013 | [-0.071, +0.033] | 0.6348 | -2.8 |  |
| full-ViT balanced (exploratory) − YOLO26m | raw | +0.061 | [+0.023, +0.090] | 0.0024 | -4.6 |  |
| full-ViT balanced (exploratory) − full-ViT | q90 | +0.044 | [+0.023, +0.067] | 0.0002 | -1.6 |  |
| full-ViT balanced (exploratory) − full-ViT | max | -0.051 | [-0.097, -0.016] | 0.0012 | -2.8 |  |
| full-ViT balanced (exploratory) − full-ViT | raw | +0.044 | [+0.023, +0.067] | 0.0002 | -1.6 |  |

## busuc:all

| Arm | rule | lesion Dice | per seed | miss % | normal FPR % |
|---|---|---:|---|---:|---:|
| full-ViT | q90 | 0.901 ± 0.005 | 0.898 / 0.899 / 0.907 | 0.3 | — |
| full-ViT | max | 0.901 ± 0.005 | 0.898 / 0.899 / 0.907 | 0.3 | — |
| full-ViT | raw | 0.901 ± 0.005 | 0.898 / 0.899 / 0.907 | 0.3 | — |
| LLM-LoRA | q90 | 0.872 ± 0.010 | 0.864 / 0.869 / 0.883 | 0.4 | — |
| LLM-LoRA | max | 0.872 ± 0.010 | 0.864 / 0.869 / 0.883 | 0.4 | — |
| LLM-LoRA | raw | 0.872 ± 0.010 | 0.864 / 0.869 / 0.883 | 0.4 | — |
| YOLO26m | q90 | 0.860 ± 0.007 | 0.869 / 0.858 / 0.854 | 2.8 | — |
| YOLO26m | max | 0.790 ± 0.018 | 0.781 / 0.777 / 0.811 | 13.1 | — |
| YOLO26m | raw | 0.861 ± 0.009 | 0.867 / 0.864 / 0.851 | 3.0 | — |
| full-ViT balanced (exploratory) | q90 | 0.905 ± 0.003 | 0.903 / 0.908 / 0.904 | 0.3 | — |
| full-ViT balanced (exploratory) | max | 0.905 ± 0.003 | 0.903 / 0.908 / 0.904 | 0.3 | — |
| full-ViT balanced (exploratory) | raw | 0.905 ± 0.003 | 0.903 / 0.908 / 0.904 | 0.3 | — |

| Comparison (patient-cluster bootstrap) | rule | dDice | 95% CI | p | dFPR | verdict |
|---|---|---:|---|---:|---:|---|
| full-ViT − YOLO26m | q90 | +0.041 | [+0.032, +0.051] | 0.0002 | — | WIN |
| full-ViT − YOLO26m | max | +0.112 | [+0.094, +0.129] | 0.0002 | — | WIN |
| full-ViT − YOLO26m | raw | +0.040 | [+0.031, +0.050] | 0.0002 | — | WIN |
| full-ViT − LLM-LoRA | q90 | +0.029 | [+0.020, +0.039] | 0.0002 | — |  |
| full-ViT − LLM-LoRA | max | +0.029 | [+0.020, +0.039] | 0.0002 | — |  |
| full-ViT − LLM-LoRA | raw | +0.029 | [+0.020, +0.039] | 0.0002 | — |  |
| LLM-LoRA − YOLO26m | q90 | +0.012 | [+0.001, +0.023] | 0.0400 | — |  |
| LLM-LoRA − YOLO26m | max | +0.082 | [+0.065, +0.100] | 0.0002 | — |  |
| LLM-LoRA − YOLO26m | raw | +0.011 | [-0.001, +0.023] | 0.0568 | — |  |
| full-ViT balanced (exploratory) − YOLO26m | q90 | +0.045 | [+0.036, +0.055] | 0.0002 | — |  |
| full-ViT balanced (exploratory) − YOLO26m | max | +0.116 | [+0.099, +0.133] | 0.0002 | — |  |
| full-ViT balanced (exploratory) − YOLO26m | raw | +0.045 | [+0.035, +0.055] | 0.0002 | — |  |
| full-ViT balanced (exploratory) − full-ViT | q90 | +0.004 | [+0.001, +0.007] | 0.0056 | — |  |
| full-ViT balanced (exploratory) − full-ViT | max | +0.004 | [+0.001, +0.007] | 0.0056 | — |  |
| full-ViT balanced (exploratory) − full-ViT | raw | +0.004 | [+0.001, +0.007] | 0.0056 | — |  |

## breast:all

| Arm | rule | lesion Dice | per seed | miss % | normal FPR % |
|---|---|---:|---|---:|---:|
| full-ViT | q90 | 0.776 ± 0.017 | 0.790 / 0.757 / 0.781 | 7.3 | 25.0 |
| full-ViT | max | 0.573 ± 0.052 | 0.543 / 0.543 / 0.634 | 34.8 | 0.0 |
| full-ViT | raw | 0.776 ± 0.017 | 0.790 / 0.757 / 0.781 | 7.3 | 25.0 |
| LLM-LoRA | q90 | 0.707 ± 0.009 | 0.703 / 0.700 / 0.717 | 9.0 | 50.0 |
| LLM-LoRA | max | 0.578 ± 0.128 | 0.666 / 0.431 / 0.635 | 29.8 | 8.3 |
| LLM-LoRA | raw | 0.707 ± 0.009 | 0.703 / 0.700 / 0.717 | 9.0 | 50.0 |
| YOLO26m | q90 | 0.713 ± 0.016 | 0.726 / 0.695 / 0.718 | 13.6 | 41.7 |
| YOLO26m | max | 0.569 ± 0.040 | 0.528 / 0.570 / 0.609 | 34.7 | 25.0 |
| YOLO26m | raw | 0.718 ± 0.006 | 0.723 / 0.719 / 0.710 | 13.2 | 41.7 |
| full-ViT balanced (exploratory) | q90 | 0.776 ± 0.008 | 0.785 / 0.774 / 0.769 | 6.9 | 25.0 |
| full-ViT balanced (exploratory) | max | 0.651 ± 0.058 | 0.587 / 0.666 / 0.701 | 24.5 | 0.0 |
| full-ViT balanced (exploratory) | raw | 0.776 ± 0.008 | 0.785 / 0.774 / 0.769 | 6.9 | 25.0 |

| Comparison (patient-cluster bootstrap) | rule | dDice | 95% CI | p | dFPR | verdict |
|---|---|---:|---|---:|---:|---|
| full-ViT − YOLO26m | q90 | +0.063 | [+0.034, +0.093] | 0.0002 | -16.7 | WIN |
| full-ViT − YOLO26m | max | +0.005 | [-0.036, +0.046] | 0.8108 | -25.0 | EQUIVALENT |
| full-ViT − YOLO26m | raw | +0.059 | [+0.031, +0.087] | 0.0002 | -16.7 | WIN |
| full-ViT − LLM-LoRA | q90 | +0.069 | [+0.044, +0.096] | 0.0002 | -25.0 |  |
| full-ViT − LLM-LoRA | max | -0.004 | [-0.037, +0.030] | 0.7996 | -8.3 |  |
| full-ViT − LLM-LoRA | raw | +0.069 | [+0.044, +0.096] | 0.0002 | -25.0 |  |
| LLM-LoRA − YOLO26m | q90 | -0.006 | [-0.041, +0.028] | 0.7120 | +8.3 |  |
| LLM-LoRA − YOLO26m | max | +0.009 | [-0.030, +0.048] | 0.6508 | -16.7 |  |
| LLM-LoRA − YOLO26m | raw | -0.011 | [-0.046, +0.023] | 0.5236 | +8.3 |  |
| full-ViT balanced (exploratory) − YOLO26m | q90 | +0.063 | [+0.033, +0.093] | 0.0002 | -16.7 |  |
| full-ViT balanced (exploratory) − YOLO26m | max | +0.082 | [+0.046, +0.119] | 0.0002 | -25.0 |  |
| full-ViT balanced (exploratory) − YOLO26m | raw | +0.058 | [+0.030, +0.086] | 0.0002 | -16.7 |  |
| full-ViT balanced (exploratory) − full-ViT | q90 | -0.000 | [-0.013, +0.012] | 0.9496 | +0.0 |  |
| full-ViT balanced (exploratory) − full-ViT | max | +0.078 | [+0.055, +0.102] | 0.0002 | +0.0 |  |
| full-ViT balanced (exploratory) − full-ViT | raw | -0.000 | [-0.013, +0.012] | 0.9496 | +0.0 |  |

## busi:test

| Arm | rule | lesion Dice | per seed | miss % | normal FPR % |
|---|---|---:|---|---:|---:|
| full-ViT | q90 | 0.815 ± 0.013 | 0.824 / 0.822 / 0.801 | 5.1 | 16.7 |
| full-ViT | max | 0.713 ± 0.066 | 0.638 / 0.741 / 0.761 | 20.2 | 0.0 |
| full-ViT | raw | 0.821 ± 0.005 | 0.826 / 0.822 / 0.817 | 2.5 | 43.3 |
| LLM-LoRA | q90 | 0.800 ± 0.007 | 0.808 / 0.797 / 0.795 | 3.5 | 33.3 |
| LLM-LoRA | max | 0.736 ± 0.067 | 0.783 / 0.660 / 0.765 | 14.6 | 3.3 |
| LLM-LoRA | raw | 0.802 ± 0.006 | 0.808 / 0.797 / 0.803 | 2.5 | 36.7 |
| YOLO26m | q90 | 0.760 ± 0.050 | 0.798 / 0.703 / 0.779 | 12.6 | 23.3 |
| YOLO26m | max | 0.711 ± 0.048 | 0.705 / 0.666 / 0.761 | 21.2 | 6.7 |
| YOLO26m | raw | 0.795 ± 0.003 | 0.799 / 0.794 / 0.793 | 6.6 | 33.3 |
| full-ViT balanced (exploratory) | q90 | 0.814 ± 0.014 | 0.826 / 0.799 / 0.816 | 3.5 | 20.0 |
| full-ViT balanced (exploratory) | max | 0.746 ± 0.076 | 0.665 / 0.758 / 0.816 | 15.2 | 6.7 |
| full-ViT balanced (exploratory) | raw | 0.816 ± 0.010 | 0.826 / 0.806 / 0.816 | 3.0 | 30.0 |

| Comparison (patient-cluster bootstrap) | rule | dDice | 95% CI | p | dFPR | verdict |
|---|---|---:|---|---:|---:|---|
| full-ViT − YOLO26m | q90 | +0.055 | [+0.020, +0.095] | 0.0002 | -6.7 | WIN |
| full-ViT − YOLO26m | max | +0.002 | [-0.042, +0.047] | 0.9692 | -6.7 | EQUIVALENT |
| full-ViT − YOLO26m | raw | +0.026 | [-0.002, +0.058] | 0.0728 | +10.0 | INCONCLUSIVE |
| full-ViT − LLM-LoRA | q90 | +0.015 | [-0.012, +0.046] | 0.2740 | -16.7 |  |
| full-ViT − LLM-LoRA | max | -0.023 | [-0.058, +0.011] | 0.1984 | -3.3 |  |
| full-ViT − LLM-LoRA | raw | +0.019 | [-0.007, +0.048] | 0.1604 | +6.7 |  |
| LLM-LoRA − YOLO26m | q90 | +0.040 | [-0.009, +0.090] | 0.1200 | +10.0 |  |
| LLM-LoRA − YOLO26m | max | +0.025 | [-0.034, +0.079] | 0.3916 | -3.3 |  |
| LLM-LoRA − YOLO26m | raw | +0.007 | [-0.036, +0.050] | 0.7804 | +3.3 |  |
| full-ViT balanced (exploratory) − YOLO26m | q90 | +0.054 | [+0.015, +0.097] | 0.0032 | -3.3 |  |
| full-ViT balanced (exploratory) − YOLO26m | max | +0.036 | [-0.009, +0.082] | 0.1188 | +0.0 |  |
| full-ViT balanced (exploratory) − YOLO26m | raw | +0.021 | [-0.010, +0.055] | 0.2072 | -3.3 |  |
| full-ViT balanced (exploratory) − full-ViT | q90 | -0.002 | [-0.017, +0.014] | 0.8472 | +3.3 |  |
| full-ViT balanced (exploratory) − full-ViT | max | +0.033 | [+0.001, +0.068] | 0.0432 | +6.7 |  |
| full-ViT balanced (exploratory) − full-ViT | raw | -0.005 | [-0.019, +0.007] | 0.4328 | -13.3 |  |

## busbra:test

| Arm | rule | lesion Dice | per seed | miss % | normal FPR % |
|---|---|---:|---|---:|---:|
| full-ViT | q90 | 0.888 ± 0.002 | 0.887 / 0.885 / 0.890 | 0.5 | — |
| full-ViT | max | 0.886 ± 0.001 | 0.886 / 0.885 / 0.888 | 0.8 | — |
| full-ViT | raw | 0.888 ± 0.002 | 0.887 / 0.885 / 0.890 | 0.5 | — |
| LLM-LoRA | q90 | 0.864 ± 0.005 | 0.867 / 0.859 / 0.867 | 0.2 | — |
| LLM-LoRA | max | 0.864 ± 0.005 | 0.867 / 0.859 / 0.867 | 0.2 | — |
| LLM-LoRA | raw | 0.864 ± 0.005 | 0.867 / 0.859 / 0.867 | 0.2 | — |
| YOLO26m | q90 | 0.891 ± 0.001 | 0.890 / 0.890 / 0.892 | 1.1 | — |
| YOLO26m | max | 0.869 ± 0.007 | 0.861 / 0.871 / 0.876 | 4.2 | — |
| YOLO26m | raw | 0.890 ± 0.001 | 0.891 / 0.889 / 0.891 | 1.2 | — |
| full-ViT balanced (exploratory) | q90 | 0.886 ± 0.004 | 0.890 / 0.884 / 0.882 | 0.7 | — |
| full-ViT balanced (exploratory) | max | 0.885 ± 0.003 | 0.888 / 0.884 / 0.882 | 0.9 | — |
| full-ViT balanced (exploratory) | raw | 0.886 ± 0.004 | 0.890 / 0.884 / 0.882 | 0.7 | — |

| Comparison (patient-cluster bootstrap) | rule | dDice | 95% CI | p | dFPR | verdict |
|---|---|---:|---|---:|---:|---|
| full-ViT − YOLO26m | q90 | -0.003 | [-0.014, +0.007] | 0.5412 | — | EQUIVALENT |
| full-ViT − YOLO26m | max | +0.017 | [+0.001, +0.033] | 0.0372 | — | WIN |
| full-ViT − YOLO26m | raw | -0.003 | [-0.014, +0.009] | 0.6416 | — | EQUIVALENT |
| full-ViT − LLM-LoRA | q90 | +0.023 | [+0.010, +0.037] | 0.0004 | — |  |
| full-ViT − LLM-LoRA | max | +0.022 | [+0.008, +0.036] | 0.0020 | — |  |
| full-ViT − LLM-LoRA | raw | +0.023 | [+0.010, +0.037] | 0.0004 | — |  |
| LLM-LoRA − YOLO26m | q90 | -0.027 | [-0.041, -0.011] | 0.0004 | — |  |
| LLM-LoRA − YOLO26m | max | -0.005 | [-0.020, +0.012] | 0.5348 | — |  |
| LLM-LoRA − YOLO26m | raw | -0.026 | [-0.040, -0.010] | 0.0004 | — |  |
| full-ViT balanced (exploratory) − YOLO26m | q90 | -0.005 | [-0.017, +0.006] | 0.3828 | — |  |
| full-ViT balanced (exploratory) − YOLO26m | max | +0.016 | [+0.000, +0.032] | 0.0488 | — |  |
| full-ViT balanced (exploratory) − YOLO26m | raw | -0.005 | [-0.016, +0.008] | 0.4600 | — |  |
| full-ViT balanced (exploratory) − full-ViT | q90 | -0.002 | [-0.007, +0.003] | 0.5104 | — |  |
| full-ViT balanced (exploratory) − full-ViT | max | -0.001 | [-0.007, +0.004] | 0.6536 | — |  |
| full-ViT balanced (exploratory) − full-ViT | raw | -0.002 | [-0.007, +0.003] | 0.5104 | — |  |

## Primary endpoint

uclm:test, full-ViT − YOLO26m, per-source q90: dDice +0.013 [-0.027, +0.051] -> **INCONCLUSIVE**. Pre-registered prediction: point estimate in [-0.05, +0.03] -> met.