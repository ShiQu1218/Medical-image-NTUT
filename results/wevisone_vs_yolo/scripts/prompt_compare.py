"""Paired comparison of two WeVisOne prompt runs on the same checkpoint (per-image flips + gate)."""
import argparse
import json
from pathlib import Path

import common
from analyze import auroc, summarize, wevis_records
from presence_experiment import gated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(common.RUNS / "wevis" / "baseline_s42"), help="original prompt run")
    ap.add_argument("--b", default=str(common.RUNS / "wevis" / "prompt_none_s42"), help="new prompt run")
    ap.add_argument("--sets", default="busi:val,busi:test,uclm:test,uclm:all")
    args = ap.parse_args()
    A, B = Path(args.a), Path(args.b)
    va, vb = wevis_records(A, "busi:val"), wevis_records(B, "busi:val")
    thr = {k: max(r["presence_score"] for r in v if r["category"] == "normal") for k, v in (("a", va), ("b", vb))}
    report = dict(val_thresholds=thr)
    lines = ["| Set | prompt | FPR % (FP/N) | lesion Dice | lesion empty % | miss % | AUROC | gate FPR % | gate lesion Dice | gate miss % |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for spec in args.sets.split(","):
        ra, rb = wevis_records(A, spec), wevis_records(B, spec)
        if not ra or not rb:
            continue
        ids = sorted({r["id"] for r in ra} & {r["id"] for r in rb})
        da, db = {r["id"]: r for r in ra}, {r["id"]: r for r in rb}
        ra, rb = [da[i] for i in ids], [db[i] for i in ids]
        out = {}
        for tag, recs, t in (("original", ra, thr["a"]), ("+None hint", rb, thr["b"])):
            s, g = summarize(recs), summarize(gated(recs, t))
            out[tag] = dict(raw=s, gate=g, auroc=auroc(recs))
            fp = round(s["fpr"] * s["normal_n"]) if s["normal_n"] else 0
            lines.append(f"| {spec} | {tag} | {100 * s['fpr']:.1f} ({fp}/{s['normal_n']}) | {s['lesion_dice']:.3f} | "
                         f"{100 * s['empty']:.1f} | {100 * s['zero_overlap']:.1f} | {auroc(recs) or float('nan'):.3f} | "
                         f"{100 * g['fpr']:.1f} | {g['lesion_dice']:.3f} | {100 * g['zero_overlap']:.1f} |")
        flips = dict(
            normal_fixed=[i for i in ids if da[i]["category"] == "normal" and da[i]["pred_pixels"] > 0 and db[i]["pred_pixels"] == 0],
            normal_broken=[i for i in ids if da[i]["category"] == "normal" and da[i]["pred_pixels"] == 0 and db[i]["pred_pixels"] > 0],
            lesion_lost=[i for i in ids if da[i]["category"] != "normal" and da[i]["intersection"] > 0 and db[i]["intersection"] == 0],
            lesion_gained=[i for i in ids if da[i]["category"] != "normal" and da[i]["intersection"] == 0 and db[i]["intersection"] > 0])
        out["flips"] = flips
        report[spec] = out
        lines.append(f"| {spec} | flips | normal fixed {len(flips['normal_fixed'])}, normal broken {len(flips['normal_broken'])}, "
                     f"lesion lost {len(flips['lesion_lost'])}, lesion gained {len(flips['lesion_gained'])} ||||||||")
    (B / "compare.json").write_text(json.dumps(report, indent=2))
    (B / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(thr), "\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
