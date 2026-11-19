"""Stratified lesion analysis + boundary metric (HD95) from cached predictions; no inference.

Models (BUSI-trained, fixed validation operating points, same as REPORT.md section 1/5b):
  WeVisOne gate (baseline_s42/43/44), YOLO val-cal (busi_m_s42/43/44), nnU-Net gate (busi_s0..s2 as available).
Lesion images only. Strata: pathology (benign/malignant) and GT area fraction at 256x256
(small < 3%, medium 3-10%, large >= 10%; fixed, label-only bins).
Per stratum: lesion Dice (misses count as 0), miss % (zero GT overlap),
HD95 in 256-grid pixels over images with a nonempty prediction (median, since HD95 is heavy-tailed).
HD95 is conditional on the model producing a mask, so a model that misses more hard lesions is scored
on an easier subset; read it together with miss %.
HD95 is in resampled pixels, not mm: cohorts have different physical spacing, so compare models
within a cohort, not cohorts with each other. mean ± SD across seeds (nnU-Net n=1).
Writes runs/comparison/STRATIFIED.md + stratified.json.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import ndimage

import analyze as A
import common

SPECS = ["busi:test", "uclm:test", "uclm:all", "busuc:all", "busbra:test", "breast:all"]
N = common.EVAL_SIZE
BINS = [("small <3%", 0, 0.03), ("medium 3-10%", 0.03, 0.10), ("large ≥10%", 0.10, 1.01)]


def unpack(bits, shape=(N, N)):
    return np.unpackbits(np.asarray(bits, np.uint8), axis=-1)[..., :shape[-1]].astype(bool).reshape(shape)


def hd95(a, b):
    """Symmetric 95th-percentile surface distance between boolean masks (both nonempty)."""
    sa = a ^ ndimage.binary_erosion(a)
    sb = b ^ ndimage.binary_erosion(b)
    da = ndimage.distance_transform_edt(~sb)[sa]
    db = ndimage.distance_transform_edt(~sa)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def gt_masks(spec):
    recs, dets = A.yolo_dets(A.YOLO_DIR / "busi_m_s42", spec)
    return {r["id"]: unpack(dets[r["id"]]["gt"]) for r in recs}


def wevis_preds(run, spec, op):
    """id -> (category, pred mask) after applying the op (gate empties masks below threshold)."""
    import json as _j
    base, filt = A._derive(spec)
    files = sorted(run.glob(f"**/{A.key(base)}_generate.jsonl"))
    raw = {}
    for line in files[0].read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = _j.loads(line)
            raw.setdefault(r["id"], r)
    ops, _ = A.model_ops("wevis", run, spec)
    out = {}
    for r in ops[op]:
        bits = raw[r["id"]].get("pred_bits")
        pred = (np.unpackbits(np.frombuffer(bytes.fromhex(bits), np.uint8))[:N * N].reshape(N, N).astype(bool)
                if bits and r["pred_pixels"] > 0 else np.zeros((N, N), bool))
        out[r["id"]] = (r["category"], pred)
    return out


def yolo_preds(run, spec):
    """YOLO val-calibrated: union of detections with conf > max(val normal max-conf)."""
    recs, dets = A.yolo_dets(run, spec)
    vrec, vdet = A.yolo_dets(run, A.yolo_val_spec(run))
    thr = max([float(vdet[r["id"]]["conf"].max()) if len(vdet[r["id"]]["conf"]) else 0.0
               for r in vrec if r["category"] == "normal"], default=0.0)
    out = {}
    for r in recs:
        d = dets[r["id"]]
        keep = d["conf"] > thr
        masks = unpack(d["masks"], (len(d["conf"]), N, N)) if len(d["conf"]) else np.zeros((0, N, N), bool)
        out[r["id"]] = (r["category"], masks[keep].any(0) if keep.any() else np.zeros((N, N), bool))
    return out


def stats(rows):
    if not rows:
        return None
    dice = [r["dice"] for r in rows]
    hd = [r["hd95"] for r in rows if r["hd95"] is not None]
    return dict(n=len(rows), lesion_dice=float(np.mean(dice)), miss=float(np.mean([r["miss"] for r in rows])),
                hd95_median=float(np.median(hd)) if hd else None, hd95_n=len(hd))


def per_image(preds, gts):
    rows = []
    for i, (cat, p) in preds.items():
        if cat == "normal":
            continue
        g = gts[i]
        inter = (p & g).sum()
        rows.append(dict(id=i, cat=cat, area=g.sum() / (N * N), dice=float(2 * inter / max(1, p.sum() + g.sum())),
                         miss=bool(inter == 0), hd95=hd95(p, g) if p.any() and g.any() else None))
    return rows


def strata(rows):
    out = {"all": stats(rows)}
    for c in ("benign", "malignant"):
        out[c] = stats([r for r in rows if r["cat"] == c])
    for name, lo, hi in BINS:
        out[name] = stats([r for r in rows if lo <= r["area"] < hi])
    return out


def agg(per_seed):
    keys = per_seed[0].keys()
    res = {}
    for k in keys:
        vals = [p[k] for p in per_seed if p[k]]
        if not vals:
            res[k] = None
            continue
        res[k] = dict(n=vals[0]["n"], **{m: A.mean_sd([v[m] for v in vals]) for m in ("lesion_dice", "miss", "hd95_median")})
    return res


def cell(s):
    if not s:
        return "–"

    def f(v, scale, nd):
        if v is None:
            return "–"
        return f"{scale * v[0]:.{nd}f}" + (f" ± {scale * v[1]:.{nd}f}" if v[2] > 1 else "")
    return f"{f(s['lesion_dice'], 1, 3)} / {f(s['miss'], 100, 1)} / {f(s['hd95_median'], 1, 1)}"


def main():
    models = [("WeVisOne gate", lambda s, sp: wevis_preds(A.WEVIS_DIR / f"baseline_s{s}", sp, "gate"), A.SEEDS),
              ("YOLO val-cal", lambda s, sp: yolo_preds(A.YOLO_DIR / f"busi_m_s{s}", sp), A.SEEDS),
              ("nnU-Net gate", lambda s, sp: wevis_preds(A.NN_RUNS[s], sp, "gate"), tuple(range(len(A.NN_RUNS))))]
    rep, md = {}, ["# Stratified lesion analysis and HD95 (BUSI-trained models, fixed val operating points)", "",
                   __doc__.split("\n", 1)[1].strip(), "",
                   "Cells: lesion Dice / miss % / median HD95 (px @256).", ""]
    for spec in SPECS:
        gts = gt_masks(spec)
        rep[spec] = {}
        raw = {}
        for name, fn, seeds in models:
            per = []
            for s in seeds:
                try:
                    rows = per_image(fn(s, spec), gts)
                except (TypeError, IndexError, FileNotFoundError):
                    continue
                raw[(name, s)] = rows
                per.append(strata(rows))
            if per:
                rep[spec][name] = agg(per)
        # paired boundary comparison on lesions BOTH models hit (same seed), removing the selection effect
        common_hd = {}
        for s in A.SEEDS:
            a, b = raw.get(("WeVisOne gate", s)), raw.get(("YOLO val-cal", s))
            if not a or not b:
                continue
            A_, B_ = {r["id"]: r for r in a}, {r["id"]: r for r in b}
            ids = [i for i in A_ if i in B_ and not A_[i]["miss"] and not B_[i]["miss"]]
            if ids:
                common_hd.setdefault("n", []).append(len(ids))
                common_hd.setdefault("wevis", []).append(float(np.median([A_[i]["hd95"] for i in ids])))
                common_hd.setdefault("yolo", []).append(float(np.median([B_[i]["hd95"] for i in ids])))
                common_hd.setdefault("wevis_dice", []).append(float(np.mean([A_[i]["dice"] for i in ids])))
                common_hd.setdefault("yolo_dice", []).append(float(np.mean([B_[i]["dice"] for i in ids])))
        if common_hd:
            rep[spec]["_common_hits"] = {k: A.mean_sd(v) for k, v in common_hd.items()}
        if not rep[spec]:
            continue
        models_rep = {k: v for k, v in rep[spec].items() if not k.startswith("_")}
        cols = list(next(iter(models_rep.values())).keys())
        first = next(iter(models_rep.values()))
        md += [f"## {spec}", "", "| Model | " + " | ".join(f"{c} (n={first[c]['n'] if first[c] else 0})" for c in cols) + " |",
               "|---|" + "---|" * len(cols)]
        for name, s in models_rep.items():
            md.append(f"| {name} | " + " | ".join(cell(s[c]) for c in cols) + " |")
        ch = rep[spec].get("_common_hits")
        if ch:
            md += ["", f"Common hits (both WeVisOne gate and YOLO val-cal overlap GT, same seed; n = {ch['n'][0]:.0f} ± {ch['n'][1]:.0f} lesions): "
                   f"median HD95 WeVisOne {ch['wevis'][0]:.1f} ± {ch['wevis'][1]:.1f} vs YOLO {ch['yolo'][0]:.1f} ± {ch['yolo'][1]:.1f} px; "
                   f"Dice {ch['wevis_dice'][0]:.3f} vs {ch['yolo_dice'][0]:.3f}."]
        md.append("")
        print(spec, "done", flush=True)
    (A.OUT / "stratified.json").write_text(json.dumps(rep, indent=2))
    (A.OUT / "STRATIFIED.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
