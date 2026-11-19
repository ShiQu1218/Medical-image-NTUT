"""Hybrid: WeVisOne decides WHETHER/WHERE (presence gate + predicted region), YOLO draws the lesion mask.

Cached predictions only, no inference. Pairs WeVisOne baseline_s{s} (gate) with YOLO busi_m_s{s}, s = 42/43/44.
YOLO caches every detection with conf >= 0.001 (max 20), with masks on the 256 grid.
Rule (prespecified, nothing tuned on test): split the gated WeVisOne mask into connected components; for each
component take the YOLO detection with the HIGHEST confidence among those whose mask has IoU >= 0.1 with the
component, and use that YOLO mask instead of the component.
  (a) W-loc -> YOLO mask, fallback WeVisOne : no matching YOLO detection -> keep the WeVisOne component
  (b) W-loc -> YOLO mask, no fallback       : no matching YOLO detection -> drop the component
  (c) (a) + YOLO val-cal detections that do not overlap any WeVisOne component (union of detectors)
Normal-image FPR follows the WeVisOne gate in (a)/(b); (c) also inherits YOLO-cal false positives.
Writes runs/comparison/HYBRID_WLOC_YSEG.md + hybrid_wloc_yseg.json
"""
from __future__ import annotations

import json

import cv2
import numpy as np

import analyze as A
import common
import stratify as S

N = S.N
SPECS = ["busi:test", "uclm:test", "uclm:all", "busuc:all", "busbra:test", "breast:all"]
IOU_MIN = 0.1


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def yolo_cal_thr(run):
    vrec, vdet = A.yolo_dets(run, A.yolo_val_spec(run))
    return max([float(vdet[r["id"]]["conf"].max()) if len(vdet[r["id"]]["conf"]) else 0.0
                for r in vrec if r["category"] == "normal"], default=0.0)


def combine(wmask, det, thr_cal):
    """Returns {variant: mask} and how many components got a YOLO mask."""
    conf = det["conf"]
    ymasks = S.unpack(det["masks"], (len(conf), N, N)) if len(conf) else np.zeros((0, N, N), bool)
    n, lab = cv2.connectedComponents(wmask.astype(np.uint8))
    a, b, used = np.zeros((N, N), bool), np.zeros((N, N), bool), 0
    for k in range(1, n):
        comp = lab == k
        cand = [(conf[j], j) for j in range(len(conf)) if iou(ymasks[j], comp) >= IOU_MIN]
        if cand:
            m = ymasks[max(cand)[1]]
            a |= m; b |= m; used += 1
        else:
            a |= comp
    c = a.copy()
    for j in range(len(conf)):
        if conf[j] > thr_cal and not (ymasks[j] & wmask).any():
            c |= ymasks[j]
    return dict(a=a, b=b, c=c), used, n - 1


def records(seed, spec, gts):
    wrun, yrun = A.WEVIS_DIR / f"baseline_s{seed}", A.YOLO_DIR / f"busi_m_s{seed}"
    wv = S.wevis_preds(wrun, spec, "gate")
    ops, _ = A.model_ops("wevis", wrun, spec)
    meta = {r["id"]: r for r in ops["gate"]}
    _, dets = A.yolo_dets(yrun, spec)
    thr = yolo_cal_thr(yrun)
    out, used, comps = {v: [] for v in "abc"}, 0, 0
    for i, (cat, wmask) in wv.items():
        g = gts[i]
        masks, u, c = combine(wmask, dets[i], thr)
        used += u; comps += c
        for v, m in masks.items():
            d, j = common.scores(m, g)
            out[v].append(dict(meta[i], pred_pixels=int(m.sum()), intersection=int((m & g).sum()), dice=d, iou=j,
                               icon_only=False, n_pred=int(m.any())))
            out[v][-1]["_mask"] = m
    return out, used / max(1, comps)


def hd95_median(recs, gts):
    v = [S.hd95(r["_mask"], gts[r["id"]]) for r in recs if r["category"] != "normal" and r["_mask"].any()]
    return float(np.median(v)) if v else None


def main():
    labels = {"wgate": "WeVisOne gate (reference)", "ycal": "YOLO val-cal (reference)",
              "a": "(a) W-loc → YOLO mask, fallback W", "b": "(b) W-loc → YOLO mask, no fallback",
              "c": "(c) (a) ∪ YOLO-cal detections"}
    rep, boot_in = {}, {}
    for spec in SPECS:
        gts = S.gt_masks(spec)
        per = {k: [] for k in labels}
        used_frac = []
        for s in A.SEEDS:
            try:
                hyb, uf = records(s, spec, gts)
            except (TypeError, FileNotFoundError):
                continue
            used_frac.append(uf)
            ref_w = A.model_ops("wevis", A.WEVIS_DIR / f"baseline_s{s}", spec)[0]["gate"]
            ref_y = A.model_ops("yolo", A.YOLO_DIR / f"busi_m_s{s}", spec)[0]["cal"]
            for k, recs in (("wgate", ref_w), ("ycal", ref_y), *hyb.items()):
                m = A.summarize([{x: v for x, v in r.items() if x != "_mask"} for r in recs])
                if k in "abc":
                    m["hd95_median"] = hd95_median(recs, gts)
                per[k].append(m)
                boot_in.setdefault((spec, k), []).append([{x: v for x, v in r.items() if x != "_mask"} for r in recs])
        if not used_frac:
            continue
        rep[spec] = {k: {m: A.mean_sd([p.get(m) for p in v]) for m in ("fpr", "lesion_dice", "zero_overlap", "hd95_median")}
                     for k, v in per.items() if v}
        rep[spec]["_yolo_mask_share"] = A.mean_sd(used_frac)
        print(spec, "done", flush=True)
    boot = {}
    for spec in ("uclm:test", "uclm:all", "busi:test"):
        for k in ("a", "b", "c"):
            if (spec, k) in boot_in and (spec, "wgate") in boot_in:
                boot[f"{spec} | {labels[k]} vs WeVisOne gate"] = A.paired_bootstrap(boot_in[(spec, k)], boot_in[(spec, "wgate")])
                boot[f"{spec} | {labels[k]} vs YOLO val-cal"] = A.paired_bootstrap(boot_in[(spec, k)], boot_in[(spec, "ycal")])
    for fam in ("lesion_dice", "fpr"):
        adj = A.holm({k: b[f"{fam}_p"] for k, b in boot.items()})
        for k, b in boot.items():
            b[f"{fam}_p_holm"] = adj.get(k, float("nan"))
    rep["paired_bootstrap"] = boot
    md = ["# Hybrid: WeVisOne localization → YOLO mask", "", __doc__.split("\n", 1)[1].strip(), "",
          "Cells: mean ± SD over seeds 42/43/44. HD95 = median over lesion images with a nonempty mask (256-grid px).", ""]
    for spec, r in rep.items():
        if spec.startswith("_") or spec == "paired_bootstrap":
            continue
        sh = r["_yolo_mask_share"]
        md += [f"## {spec}", "", f"Share of WeVisOne components replaced by a YOLO mask: {100 * sh[0]:.1f}%", "",
               "| Model | normal FPR % | lesion Dice | miss % | HD95 px |", "|---|---:|---:|---:|---:|"]
        for k, lab in labels.items():
            if k in r:
                v = r[k]
                md.append(f"| {lab} | {A.fmt(v['fpr'], True)} | {A.fmt(v['lesion_dice'])} | {A.fmt(v['zero_overlap'], True)} | "
                          f"{A.fmt(v['hd95_median']) if v.get('hd95_median') else '–'} |")
        md.append("")
    md += ["## Paired patient-cluster bootstrap (A − B; seed-averaged; Holm over this table)", "",
           "| Comparison | ΔFPR pp [95% CI] | Δ lesion Dice [95% CI] | Dice Holm p |", "|---|---:|---:|---:|"]
    for k, b in boot.items():
        f = "–" if np.isnan(b["fpr_diff"]) else f"{100 * b['fpr_diff']:+.1f} [{100 * b['fpr_diff_ci'][0]:+.1f}, {100 * b['fpr_diff_ci'][1]:+.1f}]"
        md.append(f"| {k.replace('|', '/')} | {f} | {b['lesion_dice_diff']:+.3f} [{b['lesion_dice_diff_ci'][0]:+.3f}, "
                  f"{b['lesion_dice_diff_ci'][1]:+.3f}] | {b['lesion_dice_p_holm']:.3g} |")
    (A.OUT / "hybrid_wloc_yseg.json").write_text(json.dumps(rep, indent=2, default=float))
    (A.OUT / "HYBRID_WLOC_YSEG.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
