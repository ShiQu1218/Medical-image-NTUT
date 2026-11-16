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


def is_hit(r, min_iou=None):
    return (r["pred_pixels"] > 0 and r["intersection"] > 0
            and (min_iou is None or r["iou"] >= min_iou))


def counts(recs, min_iou=None):
    tp = fn = fp = 0
    for r in recs:
        hit, pred = is_hit(r, min_iou), r["pred_pixels"] > 0
        if r["category"] == "normal":
            fp += pred
        else:
            tp += hit
            fn += not hit
            fp += pred and not hit
    return tp, fp, fn


def common_hits(a, b, min_iou=None):
    """Paired image-level union-mask hits, conditional Dice, and disjoint hit partitions."""
    aa, bb = ({r["id"]: r for r in rows} for rows in (a, b))
    if len(aa) != len(a) or len(bb) != len(b) or aa.keys() != bb.keys():
        raise ValueError("Common-hit comparisons require identical unique image IDs")
    partitions = dict(both=[], only_a=[], only_b=[], neither=[])
    for ident in sorted(aa):
        x, y = aa[ident], bb[ident]
        if x["category"] != y["category"] or x.get("patient") != y.get("patient"):
            raise ValueError(f"Inconsistent image metadata: {ident}")
        if x["category"] == "normal":
            continue
        ha, hb = is_hit(x, min_iou), is_hit(y, min_iou)
        name = "both" if ha and hb else "only_a" if ha else "only_b" if hb else "neither"
        partitions[name].append(ident)
    ids = partitions["both"]
    da = sum(aa[i]["dice"] for i in ids) / len(ids) if ids else None
    db = sum(bb[i]["dice"] for i in ids) / len(ids) if ids else None
    return dict(lesion_n=sum(map(len, partitions.values())),
                **{k: len(v) for k, v in partitions.items()}, ids=ids, partitions=partitions,
                dice_a=da, dice_b=db, dice_diff=da-db if ids else None)


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    A.OUT.mkdir(parents=True, exist_ok=True)
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
                                                                       f1_seeds=[f1(*c)[2] for c in per], counts_seeds=per)
                md.append(f"| {spec} | {name} | {op} | {len(per)} | {tp} | {fp} | {fn} | {p:.3f} | {r:.3f} | "
                          f"**{f:.3f}** | {ms[0]:.3f} ± {ms[1]:.3f} |")
    (A.OUT / "f1.json").write_text(json.dumps(out, indent=1))
    (A.OUT / "F1.md").write_text("\n".join(md), encoding="utf-8")
    strict_tables()
    common_hit_tables()
    print("\n".join(md))


CRITERIA = {"any_overlap": None, "iou_0.25": 0.25, "iou_0.50": 0.50}


def strict_tables():
    out = {}
    md = ["# Stricter image-level localized F1", "",
          "Hit = nonempty prediction with GT overlap and union-mask IoU >= the stated cutoff (256x256). "
          "TP = hit lesion image; FN = non-hit lesion image; FP = positive normal image plus nonempty non-hit lesion image. "
          "A localization failure with a nonempty prediction contributes both FP and FN. "
          "Each image contributes at most one TP. This is NOT instance-level detection F1; extra components in a hit image "
          "are penalized by union IoU, not counted as separate FP instances. All operating thresholds remain fixed from validation "
          "or the default raw/conf=0.25 rule; IoU cutoffs are evaluation criteria, not tuned operating thresholds.", "",
          "Seed-pooled counts repeat the same test images; they are not independent patients. SD is across training seeds, not a CI.", "",
          "| Set | Model | op | hit criterion | runs | TP | FP | FN | precision | recall | pooled F1 | F1 mean ± SD |",
          "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for spec in A.TESTS:
        for name, kind, runs, ops in MODELS:
            collected = [A.model_ops(kind, run, spec) for run in runs]
            for op in ops:
                for label, cutoff in CRITERIA.items():
                    per = [counts(got[0][op], cutoff) for got in collected if got and op in got[0]]
                    if not per:
                        continue
                    tp, fp, fn = (sum(c[i] for c in per) for i in range(3))
                    p, r, value = f1(tp, fp, fn)
                    seeds = [f1(*c)[2] for c in per]
                    out.setdefault(spec, {}).setdefault(name, {}).setdefault(op, {})[label] = dict(
                        n_runs=len(per), tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=value, f1_seeds=seeds,
                        counts_seeds=per)
                    ms = A.mean_sd(seeds)
                    md.append(f"| {spec} | {name} | {op} | {label} | {len(per)} | {tp} | {fp} | {fn} | "
                              f"{p:.3f} | {r:.3f} | {value:.3f} | {ms[0]:.3f} ± {ms[1]:.3f} |")
    (A.OUT / "strict_f1.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (A.OUT / "STRICT_F1.md").write_text("\n".join(md), encoding="utf-8")


def common_hit_tables():
    out = {}
    md = ["# Paired Dice on commonly hit lesion images", "",
          "A = WeVisOne baseline, B = YOLO26m-seg. Pair the same seed and identical image IDs. "
          "Report both-hit, only-A, only-B and neither-hit partitions for each criterion. Dice is averaged ONLY on both-hit "
          "lesion images within each seed, then summarized as mean ± sample SD across seeds. "
          "Counts pooled across seeds repeat images; per-seed IDs and Dice are in common_hits.json. "
          "The selected cohort depends on both models and the hit criterion; this conditional comparison cannot establish "
          "an unconditional segmentation advantage or isolate decoder quality. Use the all-lesion and GT-box oracle results alongside it. "
          "All score thresholds are validation-fixed/default, with no test-set retuning.", "",
          "| Set | A / B op | criterion | runs | both / only A / only B / neither (seed-pooled) | A Dice | B Dice | paired ΔDice |",
          "|---|---|---|---:|---|---:|---:|---:|"]
    for spec in A.TESTS:
        cached = [(s, A.model_ops("wevis", A.WEVIS_DIR / f"baseline_s{s}", spec),
                   A.model_ops("yolo", A.YOLO_DIR / f"busi_m_s{s}", spec)) for s in A.SEEDS]
        for opa, opb in (("raw", "0.25"), ("gate", "cal")):
            for label, cutoff in CRITERIA.items():
                per = [dict(seed=s, **common_hits(a[0][opa], b[0][opb], cutoff))
                       for s, a, b in cached if a and b and opa in a[0] and opb in b[0]]
                if not per:
                    continue
                stats = {k: A.mean_sd([r[k] for r in per]) for k in ("dice_a", "dice_b", "dice_diff")}
                totals = {k: sum(r[k] for r in per) for k in ("both", "only_a", "only_b", "neither", "lesion_n")}
                out.setdefault(spec, {}).setdefault(f"{opa}/{opb}", {})[label] = dict(
                    n_runs=len(per), **totals, **stats, per_seed=per)
                parts = " / ".join(str(totals[k]) for k in ("both", "only_a", "only_b", "neither"))
                md.append(f"| {spec} | {opa} / {opb} | {label} | {len(per)} | {parts} | "
                          f"{A.fmt(stats['dice_a'])} | {A.fmt(stats['dice_b'])} | {A.fmt(stats['dice_diff'])} |")
    (A.OUT / "common_hits.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (A.OUT / "COMMON_HITS.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
