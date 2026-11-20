"""Hybrid: WeVisOne decides presence + box (gate = max val-normal rule), fine-tuned MedSAM draws the mask.

Inputs: runs/medsam_ft/<set>_wevisgate_baseline_s{42,43,44}.jsonl (medsam_eval.py --modes wevis:<run>
--decoder runs/medsam_ft/best_decoder.pt). Normal-image FPR is identical to the WeVisOne gate by construction
(same images get boxes); only the mask changes. Compared with WeVisOne gate (own decoder) and YOLO val-cal.
Writes runs/comparison/WEVIS_MEDSAM.md + wevis_medsam.json
"""
from __future__ import annotations

import json

import analyze as A

SPECS = ["busi:test", "uclm:test", "uclm:all", "busuc:all", "busbra:test", "breast:all"]
FT = A.RUNS / "medsam_ft"


def main():
    labels = {"w": "WeVisOne gate (own decoder)", "h": "WeVisOne box+gate → MedSAM-ft mask", "y": "YOLO26m val-cal"}
    rep, boot_in, md = {}, {}, [__doc__.split("\n", 1)[0], "", __doc__.split("\n", 1)[1].strip(), "",
                                "| Set | Model | FPR % | lesion Dice | miss % |", "|---|---|---:|---:|---:|"]
    for spec in SPECS:
        per = {k: [] for k in labels}
        for s in A.SEEDS:
            h = A.wevis_records(FT, spec, f"wevisgate_baseline_s{s}")
            if not h:
                continue
            w = A.model_ops("wevis", A.WEVIS_DIR / f"baseline_s{s}", spec)[0]["gate"]
            y = A.model_ops("yolo", A.YOLO_DIR / f"busi_m_s{s}", spec)
            for k, recs in (("w", w), ("h", h), ("y", y[0]["cal"] if y else None)):
                if recs:
                    per[k].append(A.summarize(recs))
                    boot_in.setdefault((spec, k), []).append(recs)
        if not per["h"]:
            continue
        rep[spec] = {}
        for k, lab in labels.items():
            if per[k]:
                v = {m: A.mean_sd([p[m] for p in per[k]]) for m in ("fpr", "lesion_dice", "zero_overlap")}
                rep[spec][k] = v
                md.append(f"| {spec} | {lab} | {A.fmt(v['fpr'], True) if v['fpr'] else '–'} | {A.fmt(v['lesion_dice'])} | "
                          f"{A.fmt(v['zero_overlap'], True)} |")
    boot = {}
    for spec in SPECS:
        for b in ("w", "y"):
            if (spec, "h") in boot_in and (spec, b) in boot_in:
                boot[f"{spec} | MedSAM-ft hybrid vs {labels[b]}"] = A.paired_bootstrap(boot_in[(spec, "h")], boot_in[(spec, b)])
    adj = A.holm({k: v["lesion_dice_p"] for k, v in boot.items()})
    md += ["", "## Paired patient-cluster bootstrap (A − B, lesion Dice; Holm over this table)", "",
           "| Comparison | Δ lesion Dice [95% CI] | Holm p |", "|---|---:|---:|"]
    for k, v in boot.items():
        v["lesion_dice_p_holm"] = adj.get(k, float("nan"))
        md.append(f"| {k.replace('|', '/')} | {v['lesion_dice_diff']:+.3f} [{v['lesion_dice_diff_ci'][0]:+.3f}, "
                  f"{v['lesion_dice_diff_ci'][1]:+.3f}] | {v['lesion_dice_p_holm']:.3g} |")
    rep["paired_bootstrap"] = boot
    (A.OUT / "wevis_medsam.json").write_text(json.dumps(rep, indent=2, default=float))
    (A.OUT / "WEVIS_MEDSAM.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
