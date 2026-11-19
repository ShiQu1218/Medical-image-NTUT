"""Qualitative WeVisOne (gate) vs YOLO (val-cal) figure, seed 42, fixed validation operating points.

Cases are chosen by rule (not hand-picked), per cohort:
  WeVisOne hit / YOLO miss   : largest Dice gap in WeVisOne's favour (2 malignant + 2 benign)
  both hit                   : median-Dice common hits (2)
  WeVisOne miss / YOLO hit   : largest gap in YOLO's favour (2)  -- shown so the figure is not one-sided
  normal, false positive     : one normal with a WeVisOne FP, one with a YOLO FP (if any)
Contours on the 256x256 evaluation grid: green = GT, red = prediction. Title shows lesion Dice.
Writes runs/comparison/fig_vlm_vs_yolo_<spec>.png
"""
from __future__ import annotations

import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import analyze as A  # noqa: E402
import sources  # noqa: E402
import stratify as S  # noqa: E402

N = S.N


def pick(spec, seed=42):
    gts = S.gt_masks(spec)
    wv = S.wevis_preds(A.WEVIS_DIR / f"baseline_s{seed}", spec, "gate")
    yo = S.yolo_preds(A.YOLO_DIR / f"busi_m_s{seed}", spec)
    rows = []
    for i, (cat, pw) in wv.items():
        py = yo[i][1]
        g = gts[i]
        d = lambda p: 2 * (p & g).sum() / max(1, p.sum() + g.sum())  # noqa: E731
        rows.append(dict(id=i, cat=cat, w=pw, y=py, g=g, dw=d(pw), dy=d(py),
                         hw=bool((pw & g).any()), hy=bool((py & g).any())))
    les = [r for r in rows if r["cat"] != "normal"]
    nor = [r for r in rows if r["cat"] == "normal"]
    gap = lambda rs: sorted(rs, key=lambda r: -(r["dw"] - r["dy"]))  # noqa: E731
    am = gap([r for r in les if r["hw"] and not r["hy"] and r["cat"] == "malignant"])[:2]
    a = gap([r for r in les if r["hw"] and not r["hy"] and r["cat"] != "malignant"])[:2]
    both = sorted([r for r in les if r["hw"] and r["hy"]], key=lambda r: r["dw"])
    b = [both[len(both) // 3], both[2 * len(both) // 3]] if len(both) >= 3 else both
    c = sorted([r for r in les if r["hy"] and not r["hw"]], key=lambda r: -(r["dy"] - r["dw"]))[:2]
    fw = [r for r in nor if r["w"].any() and not r["y"].any()][:1]
    fy = [r for r in nor if r["y"].any() and not r["w"].any()][:1]
    groups = [("WeVisOne hit / YOLO miss", am), ("WeVisOne hit / YOLO miss", a), ("both hit", b), ("WeVisOne miss / YOLO hit", c),
              ("normal: WeVisOne FP", fw), ("normal: YOLO FP", fy)]
    return [(g, r) for g, rs in groups for r in rs]


def overlay(gray, gt, pred):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for m, col in ((gt, (0, 220, 0)), (pred, (255, 40, 40))):
        if m is not None and m.any():
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cs, -1, col, 2)
    return rgb


def main(spec):
    cases = pick(spec)
    want = {r["id"] for _, r in cases}
    imgs = {it["id"]: cv2.resize(np.asarray(it["image"].convert("L")), (N, N), interpolation=cv2.INTER_AREA)
            for it in sources.items(spec) if it["id"] in want}
    fig, axes = plt.subplots(len(cases), 3, figsize=(7.5, 2.6 * len(cases)))
    axes = np.atleast_2d(axes)
    for k, (group, r) in enumerate(cases):
        gray = imgs[r["id"]]
        lesion = r["cat"] != "normal"
        panels = [(f"{r['cat']}  [{group}]", overlay(gray, r["g"], None)),
                  ("WeVisOne" + (f"  Dice {r['dw']:.2f}" if lesion else ("  FP" if r["w"].any() else "  clean")),
                   overlay(gray, r["g"], r["w"])),
                  ("YOLO" + (f"  Dice {r['dy']:.2f}" if lesion else ("  FP" if r["y"].any() else "  clean")),
                   overlay(gray, r["g"], r["y"]))]
        for ax, (t, im) in zip(axes[k], panels):
            ax.imshow(im)
            ax.set_title(t, fontsize=8)
            ax.axis("off")
    fig.suptitle(f"{spec}: WeVisOne gate vs YOLO26m val-cal (seed 42)  green = GT, red = prediction", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / len(cases)))
    out = A.OUT / f"fig_vlm_vs_yolo_{spec.replace(':', '_')}.png"
    fig.savefig(out, dpi=130)
    print(out, len(cases), "cases")


if __name__ == "__main__":
    for spec in sys.argv[1:] or ["uclm:test", "busi:test"]:
        main(spec)
