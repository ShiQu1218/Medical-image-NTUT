"""Training curves from history.csv + the learning-rate schedule.

LR was not logged; it is reconstructed exactly from train.py (deterministic): AdamW groups LoRA lr_lora and decoder
lr_decoder, LambdaLR factor(s) = min(1, (s+1)/warmup) * 0.5 * (1 + cos(pi * min(1, s/total))), warmup = max(1, int(0.05*total)),
s = number of completed scheduler steps (update k uses factor(k-1)). total and the base LRs come from each run's args.json.
Loss points = mean over the samples since the previous log (every 5 updates); thick line = moving average of 5 points.
Usage: plot_history.py [group ...]
"""
import csv
import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
import sources  # noqa: E402

R = common.RUNS
GROUPS = {"baseline": [R / "lora_decoder", R / "sweep/A_seed43", R / "sweep/A_seed44"],
          "aug": [R / f"wevis/aug_s{s}" for s in (42, 43, 44)],
          "ms_bra_uclm": [R / f"wevis/ms_bra_uclm_s{s}" for s in (42, 43, 44)]}


def load(run):
    rows = list(csv.DictReader((run / "history.csv").open(encoding="utf-8")))
    return [r for r in rows if r["lm_loss"]], [r for r in rows if r["val_dice"]]


def lr_curve(run):
    a = json.loads((run / "args.json").read_text())
    n_train = len(sources.rows(a.get("train_spec", "busi:train")))
    spe = math.ceil(n_train / a["accum"])
    total = a.get("max_steps") or a.get("total_steps") or spe * a["epochs"]
    warm = max(1, int(0.05 * total))
    k = np.arange(1, total + 1)
    s = k - 1
    f = np.minimum(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * np.minimum(1.0, s / total)))
    return k, f * a["lr_lora"], f * a["lr_decoder"], a.get("val_spec", "busi:val")


def smooth(y, w=5):
    y = np.asarray(y, float)
    return np.convolve(y, np.ones(w) / w, mode="valid"), (w - 1) // 2


for g in sys.argv[1:] or ["baseline"]:
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for run, s, c in zip(GROUPS[g], (42, 43, 44), ("C0", "C1", "C2")):
        tr, va = load(run)
        x = np.array([int(r["step"]) for r in tr])
        for i, key in ((0, "lm_loss"), (1, "seg_loss")):
            y = [float(r[key]) for r in tr]
            ax[i].plot(x, y, color=c, alpha=.3, lw=.8)
            ys, off = smooth(y)
            ax[i].plot(x[off:off + len(ys)], ys, color=c, lw=2, label=f"seed {s}")
        xv = [int(r["step"]) for r in va]
        ax[2].plot(xv, [float(r["val_lesion_dice"]) for r in va], "o-", color=c, label=f"lesion Dice s{s}")
        ax[2].plot(xv, [float(r["val_normal_fp"]) for r in va], "x--", color=c, alpha=.6, label=f"normal FP s{s}")
        if s == 42:
            k, lr_l, lr_d, vs = lr_curve(run)
            ax[3].plot(k, lr_l, color="C3", label="LoRA (base 1e-4)")
            ax[3].plot(k, lr_d, color="C4", label="decoder (base 3e-5)")
            best = json.loads((run / "best.json").read_text())["step"]
            for a_ in ax:
                a_.axvline(best, color="k", ls=":", lw=.8)
    ax[0].set(title="LM loss (train)", xlabel="optimizer update", yscale="log")
    ax[1].set(title="Seg loss (train, BCE+Dice x3 scales)", xlabel="optimizer update")
    ax[2].set(title=f"Validation ({vs}, generate)", xlabel="optimizer update", ylim=(0, 1))
    ax[2].title.set_fontsize(9)
    ax[3].set(title="Learning rate (reconstructed, all seeds identical)", xlabel="optimizer update")
    ax[3].title.set_fontsize(9)
    ax[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for a_ in ax:
        a_.grid(alpha=.3)
        a_.legend(fontsize=7)
    fig.suptitle(f"WeVisOne LoRA + decoder training: {g}   (dotted line = checkpoint selected on val, seed 42)")
    fig.tight_layout()
    out = R / "comparison" / f"fig_training_lr_{g}.png"
    fig.savefig(out, dpi=120)
    print(out)
