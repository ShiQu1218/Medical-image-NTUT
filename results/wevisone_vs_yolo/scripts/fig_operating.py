"""Operating curves on BUS-UCLM: lesion Dice and zero-overlap rate vs normal-image FPR, every threshold.

Writes runs/comparison/fig_operating_uclm.png. Thin line per seed; colour + marker identify the model.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import analyze as A  # noqa: E402

MODELS = [("WeVisOne-VL-4B", "wevis", [A.WEVIS_DIR / f"baseline_s{s}" for s in A.SEEDS], "#2a78d6", "o"),
          ("YOLO26m-seg", "yolo", [A.YOLO_DIR / f"busi_m_s{s}" for s in A.SEEDS], "#eb6834", "s"),
          ("nnU-Net v2 2d", "wevis", [A.RUNS / "nnunet" / "busi_s0"], "#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    specs = [("uclm:all", "BUS-UCLM all (419 normal / 260 lesion)"), ("uclm:test", "BUS-UCLM held-out patients (188 / 143)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, facecolor="#fcfcfb")
    for col, (spec, title) in enumerate(specs):
        for name, kind, runs, color, marker in MODELS:
            for i, run in enumerate(runs):
                pts = [p for p in (A.operating_curve(kind, run, spec) or []) if p[0] is not None]
                pts.sort(key=lambda p: p[0])
                x = [100 * p[0] for p in pts]
                for row, j, scale in ((0, 1, 1), (1, 2, 100)):
                    ax = axes[row, col]
                    ax.plot(x, [scale * p[j] for p in pts], color=color, lw=2 if i == 0 else 1.2,
                            alpha=1 if i == 0 else 0.55, label=name if i == 0 else None,
                            marker=marker, markevery=max(1, len(pts) // 8), ms=8 if i == 0 else 0,
                            markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        axes[0, col].set_title(title, color=INK, fontsize=11, loc="left")
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")
        ax.grid(color=GRID, lw=0.8)
        ax.axvline(10, color=MUTED, lw=1, ls="--")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)
        ax.set_xlim(0, 100)
    axes[0, 0].set_ylabel("Lesion Dice (per-image mean)", color=INK)
    axes[1, 0].set_ylabel("Lesion zero-overlap rate (%)", color=INK)
    for ax in axes[1]:
        ax.set_xlabel("Normal-image false-positive rate (%)", color=INK)
    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.suptitle("Trained on BUSI only. Every threshold of each model's image-level score; bold = seed 42, thin = seeds 43/44; dashed = FPR 10%",
                 color=MUTED, fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = A.OUT / "fig_operating_uclm.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print("wrote", out)


if __name__ == "__main__":
    main()
