"""Decompose end-to-end lesion failures (reviewer item 2): where does generate-mode Dice go?

Per lesion image, from the generated text (WeVisOne) or kept detections (YOLO):
  none       : no box at all (answered None / no detection)
  malformed  : box text present but no mask decoded (e.g. box without <OBJ>)
  off        : best box IoU with any GT box < 0.1   (wrong place)
  partial    : 0.1 <= best IoU < 0.5
  hit        : best IoU >= 0.5
Also: box-level recall (GT boxes matched at IoU >= 0.5), precision (pred boxes matched), count errors.
GT boxes = tight boxes of GT instances (connected components where instance masks are unavailable).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import common
import sources
from analyze import RUNS, wevis_records, yolo_dets, uclm_test_ids
from yolo_eval import dets_to_records

BOX = re.compile(r"<(\d+)><(\d+)><(\d+)><(\d+)>")
CATS = ["none", "malformed", "off", "partial", "hit"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


_GT = {}


def gt_boxes(spec):
    base = "uclm:all" if spec == "uclm:test" else spec
    if base not in _GT:
        _GT[base] = {it["id"]: [common.tight_box(m) for m in it["instances"]] for it in sources.items(base)}
    return _GT[base]


def classify(recs, pred_boxes, gtb):
    rows = []
    for r in recs:
        if r["category"] == "normal":
            continue
        pb, gb = pred_boxes(r), gtb[r["id"]]
        best = max((iou(p, g) for p in pb for g in gb), default=0.0)
        if not pb:
            cat = "none"
        elif r["n_pred"] == 0 or r["pred_pixels"] == 0 and best >= 0.1:
            cat = "malformed"
        else:
            cat = "off" if best < 0.1 else "partial" if best < 0.5 else "hit"
        rows.append(dict(id=r["id"], cat=cat, best_iou=best, dice=r["dice"], n_pred=len(pb), n_gt=len(gb),
                         gt_matched=sum(any(iou(p, g) >= 0.5 for p in pb) for g in gb),
                         pred_matched=sum(any(iou(p, g) >= 0.5 for g in gb) for p in pb)))
    return rows


def summarize(rows):
    n = len(rows)
    out = dict(n=n)
    for c in CATS:
        sub = [r for r in rows if r["cat"] == c]
        out[c] = dict(k=len(sub), pct=len(sub) / n if n else None,
                      mean_dice=float(np.mean([r["dice"] for r in sub])) if sub else None,
                      dice_lost=float(sum(1 - r["dice"] for r in sub) / n) if n else None)
    gt_tot = sum(r["n_gt"] for r in rows)
    pr_tot = sum(r["n_pred"] for r in rows)
    out["box_recall@0.5"] = sum(r["gt_matched"] for r in rows) / gt_tot if gt_tot else None
    out["box_precision@0.5"] = sum(r["pred_matched"] for r in rows) / pr_tot if pr_tot else None
    out["count_error_rate"] = float(np.mean([r["n_pred"] != r["n_gt"] for r in rows])) if n else None
    out["lesion_dice"] = float(np.mean([r["dice"] for r in rows])) if n else None
    return out


def wevis_boxes(r):
    return [[int(v) / 1000 for v in m] for m in BOX.findall(r.get("response") or "")]


def main():
    out, md = {}, ["# Failure decomposition of end-to-end lesion predictions", "",
                   "Category share of lesion images (k/N) and mean Dice inside each category. "
                   "`dice lost` = that category's share of the total (1 − Dice) budget per image, so the columns add up to 1 − lesion Dice.", "",
                   "| Set | Model | N | none | malformed | off (IoU<0.1) | partial | hit (IoU≥0.5) | box recall@0.5 | box precision@0.5 | count err | lesion Dice |",
                   "|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|"]
    specs = ["busi:test", "uclm:test", "uclm:all", "busuc:all", "busbra:test", "breast:all"]
    for spec in specs:
        gtb = gt_boxes(spec)
        models = []
        for s in (42, 43, 44):
            recs = wevis_records(RUNS / "wevis" / f"baseline_s{s}", spec)
            if recs:
                models.append((f"WeVisOne s{s}", recs, wevis_boxes))
        for s in (42, 43, 44):
            run = RUNS / "yolo" / f"busi_m_s{s}"
            got = yolo_dets(run, spec)
            if not got:
                continue
            recs, dets = got
            thr = json.loads((run / "eval" / "report.json").read_text())["calibrated_threshold"]
            for tag, t, strict in (("cal", thr, True), ("0.25", 0.25, False)):
                rr = dets_to_records(recs, dets, t, strict=strict)
                keep = {r["id"]: [list(map(float, b)) for b, c in zip(dets[r["id"]]["boxes"], dets[r["id"]]["conf"])
                                  if (c > t if strict else c >= t)] for r in recs}
                models.append((f"YOLO s{s} {tag}", rr, lambda r, keep=keep: keep[r["id"]]))
        for name, recs, fn in models:
            rows = classify(recs, fn, gtb)
            if not rows:
                continue
            s = summarize(rows)
            out[f"{spec} | {name}"] = dict(summary=s, per_image=rows)
            cell = lambda c: f"{s[c]['k']}/{s['n']} ({s[c]['mean_dice']:.2f}; lost {s[c]['dice_lost']:.3f})" if s[c]["k"] else "0"  # noqa: E731
            md.append(f"| {spec} | {name} | {s['n']} | " + " | ".join(cell(c) for c in CATS) +
                      f" | {s['box_recall@0.5']:.3f} | {s['box_precision@0.5'] or 0:.3f} | {s['count_error_rate']:.3f} | {s['lesion_dice']:.3f} |")
    (RUNS / "comparison").mkdir(exist_ok=True)
    (RUNS / "comparison" / "failures.json").write_text(json.dumps(out, indent=1))
    (RUNS / "comparison" / "FAILURES.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
