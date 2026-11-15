"""Image-level localized F1 for every model / operating point / test set (k/N pooled over seeds).

TP = lesion image whose prediction overlaps the GT.  FN = lesion image with zero overlap (no or wrong prediction).
FP = normal image with any predicted pixel + lesion image whose prediction misses the GT entirely.
Pixel-level F1 equals Dice and is the 'lesion Dice' column elsewhere. Writes runs/comparison/F1.md and f1.json.
"""
from __future__ import annotations

import json

import analyze as A

MODELS = [("WeVisOne", "wevis", [A.WEVIS_DIR / f"baseline_s{s}" for s in A.SEEDS], ("raw", "gate")),
          ("WeVisOne strong-aug", "wevis", [A.WEVIS_DIR / f"aug_s{s}" for s in A.SEEDS], ("raw", "gate")),
          ("YOLO26m-seg", "yolo", [A.YOLO_DIR / f"busi_m_s{s}" for s in A.SEEDS], ("0.25", "cal")),
          ("nnU-Net v2 2d", "wevis", [A.RUNS / "nnunet" / "busi_s0"], ("raw", "gate"))]


def counts(recs):
    tp = fn = fp = 0
    for r in recs:
        hit, pred = r["intersection"] > 0, r["pred_pixels"] > 0
        if r["category"] == "normal":
            fp += pred
        else:
            tp += hit
            fn += not hit
            fp += pred and not hit
    return tp, fp, fn


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    out = {}
    md = ["# Image-level localized F1", "", __doc__.split("\n\n")[1].replace("\n", " "), "",
          "| Set | Model | op | runs | TP | FP | FN | precision | recall | F1 (pooled) | F1 mean ± SD over seeds |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for spec in A.TESTS:
        for name, kind, runs, ops in MODELS:
            for op in ops:
                per = []
                for run in runs:
                    got = A.model_ops(kind, run, spec)
                    if got and op in got[0]:
                        per.append(counts(got[0][op]))
                if not per:
                    continue
                tp, fp, fn = (sum(c[i] for c in per) for i in range(3))
                p, r, f = f1(tp, fp, fn)
                ms = A.mean_sd([f1(*c)[2] for c in per])
                out.setdefault(spec, {}).setdefault(name, {})[op] = dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f,
                                                                       f1_seeds=[f1(*c)[2] for c in per])
                md.append(f"| {spec} | {name} | {op} | {len(per)} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | "
                          f"**{f:.3f}** | {ms[0]:.3f} ± {ms[1]:.3f} |")
    (A.OUT / "f1.json").write_text(json.dumps(out, indent=1))
    (A.OUT / "F1.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
