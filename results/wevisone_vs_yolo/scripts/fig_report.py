"""Figures for the summary PDF (cached scores only, no inference).

Writes runs/comparison/fig_report_*.png and fig_report.json (every plotted number, for the PDF tables).
Gate rules are fitted on each run's own val split (see gate_rules.py): 'max' = max val-normal score, 'q90' = 90th pct.
Palette: validated default categorical slots 1-4 (dataviz skill), fixed per entity across all figures.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze as A
from gate_rules import rules
from presence_experiment import gated

plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"], "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
C = dict(lora="#2a78d6", yolo="#eb6834", vitfull="#1baf7a", medsam="#eda100", other="#a3a29c")
INK, INK2 = "#0b0b0b", "#52514e"
SPECS = ["busi:test", "uclm:test", "uclm:all", "busuc:all", "busbra:test", "breast:all"]
SET_LABEL = {"busi:test": "BUSI test\n(同院)", "uclm:test": "UCLM test", "uclm:all": "UCLM 全部", "busuc:all": "BUS_UC",
             "busbra:test": "BUS-BRA test", "breast:all": "BrEaST"}
OUT = A.OUT


def wevis_arm(name, rule):
    """{spec: [summary per seed]} for a WeVisOne run family under a val-fitted gate rule."""
    res = {}
    for s in A.SEEDS:
        run = A.WEVIS_DIR / f"{name}_s{s}"
        t = rules(A.wevis_records(run, A.wevis_val_spec(run)))[0][rule]
        for spec in SPECS:
            recs = A.wevis_records(run, spec)
            if recs is not None:
                res.setdefault(spec, []).append(A.summarize(gated(recs, t)))
    return res


def yolo_arm(name="busi_m", rule="max"):
    res = {}
    for s in A.SEEDS:
        run = A.YOLO_DIR / f"{name}_s{s}"
        vrec, vdet = A.yolo_dets(run, A.yolo_val_spec(run))
        t = rules(A.dets_to_records(vrec, vdet, 1.0))[0][rule]
        for spec in SPECS:
            got = A.yolo_dets(run, spec)
            if got is not None:
                res.setdefault(spec, []).append(A.summarize(A.dets_to_records(*got, t)))
    return res


def ms(arm, spec, m):
    v = [x[m] for x in arm.get(spec, []) if x[m] is not None]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, [float(x) for x in v]) if v else None


def save(fig, name):
    fig.savefig(OUT / f"fig_report_{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_main(arms, data):
    """Grouped bars: lesion Dice per set for the three headline models."""
    keys = [("yolo", "YOLO26m（val 校準）"), ("lora", "WeVisOne LLM LoRA（max 門檻）"), ("vitfull", "WeVisOne 全量 ViT（q90 門檻）")]
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    x, w = np.arange(len(SPECS)), 0.26
    for i, (k, lab) in enumerate(keys):
        vals = [ms(arms[k], s, "lesion_dice") for s in SPECS]
        ax.bar(x + (i - 1) * w, [v[0] for v in vals], w - 0.03, yerr=[v[1] for v in vals], color=C[k], label=lab,
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=2), zorder=2)
        if k == "vitfull":
            for xi, v in zip(x, vals):
                ax.text(xi + w, v[0] + v[1] + 0.015, f"{v[0]:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        data.setdefault("main", {})[k] = {s: v for s, v in zip(SPECS, vals)}
    ax.set_xticks(x, [SET_LABEL[s] for s in SPECS])
    ax.set_ylabel("病灶 Dice（含漏判 = 0）")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=8)
    save(fig, "main")


OFFS = {"lora": (-7, -6, "right"), "lora_q90": (7, 6, "left"), "vitfull": (7, -6, "left"), "aug": (-7, -4, "right"),
        "long690": (0, 9, "center"), "lokr": (-7, 4, "right"), "yolo_aug": (-7, 3, "right"), "vitlora": (7, 3, "left"),
        "vitfull_max": (7, 3, "left")}


def fig_tradeoff(arms, data):
    """UCLM test: normal-image FPR vs lesion miss for every training variant."""
    pts = [("yolo", "YOLO26m", C["yolo"]), ("yolo_x", "YOLO26x", C["yolo"]), ("yolo_aug", "YOLO26m + 強增強", C["yolo"]),
           ("lora", "LLM LoRA（max）", C["lora"]), ("lora_q90", "LLM LoRA（q90）", C["lora"]),
           ("aug", "強增強", C["other"]), ("long690", "690 步", C["other"]), ("lokr", "LoKr", C["other"]),
           ("lokr_am3", "LoKr+Automagic3", C["other"]), ("vitlora", "ViT LoRA", C["other"]),
           ("vitfull_max", "全量 ViT（max）", C["vitfull"]), ("vitfull", "全量 ViT（q90）", C["vitfull"])]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    rows = {}
    for k, lab, col in pts:
        f, z, d = ms(arms[k], "uclm:test", "fpr"), ms(arms[k], "uclm:test", "zero_overlap"), ms(arms[k], "uclm:test", "lesion_dice")
        rows[lab] = dict(fpr=f, miss=z, dice=d)
        hollow = k in ("lora_q90", "vitfull_max", "yolo_x", "yolo_aug")  # filled = YOLO26m, original main, new main, others
        ax.scatter(100 * f[0], 100 * z[0], s=46, zorder=3, color="#fcfcfb" if hollow else col, edgecolor=col, linewidth=1.6)
        dx, dy, ha = OFFS.get(k, (6, 2, "left"))
        ax.annotate(lab, (100 * f[0], 100 * z[0]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, ha=ha,
                    va="center", color=INK)
    ax.set_xlabel("正常影像誤報率 %（UCLM test，188 張正常）")
    ax.set_ylabel("病灶漏判率 %（429 張病灶）")
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    ax.text(0.99, 0.03, "實心 = 主要比較的設定；空心 = 同家族的變體\n顏色：橘 YOLO、藍 LLM LoRA、綠 全量 ViT、灰 其他 WeVisOne 訓練變體",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.text(0.99, 0.98, "越靠左下越好", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    data["tradeoff_uclm_test"] = rows
    save(fig, "tradeoff")


def fig_gate(arms, data):
    """Per-seed UCLM test lesion Dice under the max vs q90 rule (dumbbell)."""
    fams = [("LLM LoRA", "lora", "lora_q90", C["lora"]), ("全量 ViT", "vitfull_max", "vitfull", C["vitfull"]),
            ("BUSI+BRA LLM LoRA", "msbra", "msbra_q90", C["lora"]), ("BUSI+BRA 全量 ViT", "msbra_vitfull_max", "msbra_vitfull", C["vitfull"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    rec = {}
    for ax, spec in zip(axes, ("uclm:test", "breast:all")):
        for i, (lab, kmax, kq, col) in enumerate(fams):
            a, b = ms(arms[kmax], spec, "lesion_dice")[2], ms(arms[kq], spec, "lesion_dice")[2]
            rec.setdefault(spec, {})[lab] = dict(max=a, q90=b)
            for j, (u, v) in enumerate(zip(a, b)):
                y = i + (j - 1) * 0.22
                ax.plot([u, v], [y, y], color=col, lw=1.4, alpha=0.6, zorder=2)
                ax.scatter(u, y, s=30, color="#fcfcfb", edgecolor=col, linewidth=1.5, zorder=3)
                ax.scatter(v, y, s=30, color=col, zorder=3)
        ax.set_title(SET_LABEL[spec].replace("\n", " "), fontsize=9, color=INK)
        ax.set_xlabel("病灶 Dice")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(fams)), [f[0] for f in fams])
    axes[0].invert_yaxis()
    h1 = axes[1].scatter([], [], s=30, color="#fcfcfb", edgecolor=INK2, linewidth=1.5, label="max 門檻（13 張 val 正常影像最高分）")
    h2 = axes[1].scatter([], [], s=30, color=INK2, label="q90 門檻（第 90 百分位）")
    fig.legend(handles=[h1, h2], loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=8)
    fig.text(0.5, -0.09, "每條線 = 一個訓練 seed（42/43/44，由上到下）", ha="center", fontsize=8, color=INK2)
    data["gate_per_seed"] = rec
    save(fig, "gate")


def fig_forest(data):
    boot = json.loads((OUT / "gate_q90_bootstrap.json").read_text())
    groups = [("full-ViT q90 vs LLM-LoRA q90", "只用 BUSI 訓練：全量 ViT − LLM LoRA（皆 q90）"),
              ("BUSI+BRA full-ViT q90 vs BUSI+BRA LLM-LoRA q90", "BUSI+BRA 訓練：全量 ViT − LLM LoRA（皆 q90）")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    rec = {}
    for ax, (key, title) in zip(axes, groups):
        for i, spec in enumerate(SPECS):
            v = boot[f"{spec} | {key}"]
            lo, hi, d, p = *v["lesion_dice_diff_ci"], v["lesion_dice_diff"], v["lesion_dice_p_holm"]
            sig = p < 0.05
            ax.plot([lo, hi], [i, i], color=C["vitfull"] if sig else C["other"], lw=2, solid_capstyle="round", zorder=2)
            ax.scatter(d, i, s=34, color=C["vitfull"] if sig else "#fcfcfb", edgecolor=C["vitfull"] if sig else C["other"],
                       linewidth=1.5, zorder=3)
            ax.annotate(f"{d:+.3f}" + ("*" if sig else ""), (hi, i), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK)
            rec.setdefault(key, {})[spec] = dict(diff=d, ci=[lo, hi], p_holm=p)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_xlabel("病灶 Dice 差（95% CI）")
        ax.set_xlim(-0.06, 0.14)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(SPECS)), [SET_LABEL[s].replace("\n", " ") for s in SPECS])
    axes[0].invert_yaxis()
    fig.text(0.5, -0.09, "實心 + * = Holm 校正後 p < 0.05（30 個比較一起校正）；病人群集 bootstrap 5,000 次",
             ha="center", fontsize=8, color=INK2)
    data["forest"] = rec
    save(fig, "forest")


def fig_variants(arms, data):
    """UCLM test lesion Dice and FPR for each WeVisOne training variant (two panels, same row order)."""
    rows = [("lora", "LLM LoRA 230 步（原主方法）", C["lora"]), ("aug", "+ 強增強", C["other"]),
            ("long460", "460 步", C["other"]), ("long690", "690 步", C["other"]), ("lokr", "LoKr（AdamW）", C["other"]),
            ("lokr_am3", "LoKr + Automagic3", C["other"]), ("vitlora", "LLM + ViT LoRA", C["other"]),
            ("vitfull_max", "全量 ViT（max 門檻）", C["vitfull"]), ("vitfull", "全量 ViT（q90 門檻）", C["vitfull"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    rec = {}
    for ax, (m, xl, scale) in zip(axes, (("lesion_dice", "病灶 Dice", 1), ("fpr", "正常影像誤報率 %", 100))):
        for i, (k, lab, col) in enumerate(rows):
            v = ms(arms[k], "uclm:test", m)
            rec.setdefault(lab, {})[m] = v
            ax.errorbar(scale * v[0], i, xerr=scale * v[1], fmt="o", ms=6, color=col, ecolor=col, elinewidth=1.2,
                        capsize=2, mfc=col if k != "vitfull_max" else "#fcfcfb", mew=1.5, zorder=3)
            ax.annotate(f"{scale * v[0]:.3f}" if scale == 1 else f"{scale * v[0]:.1f}", (scale * (v[0] + v[1]), i),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.2, color=INK)
        base = scale * ms(arms["lora"], "uclm:test", m)[0]
        ax.axvline(base, color=C["lora"], lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_xlabel(xl + "（UCLM test）")
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(0.45, 0.9)
    axes[1].set_xlim(0, 65)
    axes[0].set_yticks(range(len(rows)), [r[1] for r in rows])
    axes[0].invert_yaxis()
    fig.text(0.5, -0.08, "點 = 3 個 seed 平均，橫線 = ±1 SD；虛線 = 原主方法", ha="center", fontsize=8, color=INK2)
    data["variants_uclm_test"] = rec
    save(fig, "variants")


def fig_decoder(data):
    rep = json.loads((OUT / "wevis_medsam.json").read_text())
    keys = [("y", "YOLO26m（val 校準）", C["yolo"]), ("h", "WeVisOne 框 → 微調 MedSAM 畫 mask", C["medsam"]),
            ("w", "WeVisOne 自己的 decoder", C["lora"])]
    specs = [s for s in SPECS if s in rep]
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    x, w = np.arange(len(specs)), 0.26
    rec = {}
    for i, (k, lab, col) in enumerate(keys):
        vals = [rep[s][k]["lesion_dice"] for s in specs]
        mean = [v[0] if isinstance(v, (list, tuple)) else v["mean"] for v in vals]
        sd = [v[1] if isinstance(v, (list, tuple)) else v["sd"] for v in vals]
        ax.bar(x + (i - 1) * w, mean, w - 0.03, yerr=sd, color=col, label=lab, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        rec[k] = dict(zip(specs, mean))
    ax.set_xticks(x, [SET_LABEL[s] for s in specs])
    ax.set_ylabel("病灶 Dice")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False, fontsize=8)
    data["decoder"] = rec
    save(fig, "decoder")


def main():
    arms = dict(lora=wevis_arm("baseline", "max"), lora_q90=wevis_arm("baseline", "q90"),
                vitfull=wevis_arm("vitfull", "q90"), vitfull_max=wevis_arm("vitfull", "max"),
                aug=wevis_arm("aug", "max"), long460=wevis_arm("long460", "max"), long690=wevis_arm("long690", "max"),
                lokr=wevis_arm("lokr", "max"), lokr_am3=wevis_arm("lokr_am3", "max"), vitlora=wevis_arm("vitlora", "max"),
                msbra=wevis_arm("ms_bra", "max"), msbra_q90=wevis_arm("ms_bra", "q90"),
                msbra_vitfull=wevis_arm("ms_bra_vitfull", "q90"), msbra_vitfull_max=wevis_arm("ms_bra_vitfull", "max"),
                yolo=yolo_arm("busi_m"), yolo_x=yolo_arm("busi_x"), yolo_aug=yolo_arm("busi_maug"))
    data = {}
    fig_main(arms, data)
    fig_tradeoff(arms, data)
    fig_gate(arms, data)
    fig_forest(data)
    fig_variants(arms, data)
    fig_decoder(data)
    data["all"] = {k: {s: {m: ms(v, s, m) for m in ("fpr", "lesion_dice", "zero_overlap")} for s in SPECS} for k, v in arms.items()}
    (OUT / "fig_report.json").write_text(json.dumps(data, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
