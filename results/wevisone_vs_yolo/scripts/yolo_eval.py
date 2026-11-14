"""Run a YOLO-seg checkpoint once at very low confidence and keep every detection.

Per item we save all detection confidences and their 256x256 masks, so any threshold
(default 0.25, val-calibrated, ROC sweeps) can be recomputed offline with dets_to_records().
Presence score = max detection confidence (0 when nothing is detected).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import common
from fpr_metrics import metrics
import sources


def predict_items(model, items, imgsz=640):
    """items: iterable of dict(id, category, image(PIL), union(np), **meta). Returns (records, dets)."""
    records, dets = [], {}
    for k, it in enumerate(items):
        bgr = cv2.cvtColor(np.asarray(it["image"].convert("RGB")), cv2.COLOR_RGB2BGR)
        res = model.predict(bgr, conf=0.001, iou=0.7, max_det=20, imgsz=imgsz, retina_masks=True, verbose=False)[0]
        confs = res.boxes.conf.cpu().numpy().astype(np.float32) if res.boxes is not None else np.zeros(0, np.float32)
        boxes = res.boxes.xyxyn.cpu().numpy().astype(np.float32) if res.boxes is not None else np.zeros((0, 4), np.float32)
        if res.masks is not None and len(confs):
            m = res.masks.data.cpu().numpy().astype(np.float32)
            m256 = np.stack([cv2.resize(x, (common.EVAL_SIZE,) * 2, interpolation=cv2.INTER_LINEAR) > 0.5 for x in m])
        else:
            m256 = np.zeros((0, common.EVAL_SIZE, common.EVAL_SIZE), bool)
            confs, boxes = confs[:0], boxes[:0]
        gt = cv2.resize(it["union"], (common.EVAL_SIZE,) * 2, interpolation=cv2.INTER_NEAREST)
        dets[it["id"]] = dict(conf=confs, boxes=boxes, masks=np.packbits(m256, axis=-1),
                              gt=np.packbits(gt.astype(bool), axis=-1))
        records.append({k2: v for k2, v in it.items() if k2 not in ("image", "union", "instances")})
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}", flush=True)
    return records, dets


def icon_only(pred: np.ndarray) -> bool:
    """Prediction lies entirely in the bottom-right corner (body-mark pictogram area, as on BUS-UCLM)."""
    ys, xs = np.where(pred)
    return bool(len(ys)) and ys.min() >= 0.80 * pred.shape[0] and xs.min() >= 0.85 * pred.shape[1]


def dets_to_records(records, dets, threshold, strict=True, drop_icon=False):
    """Keep detections with conf > threshold (strict) or >= threshold; union them; score like WeVisOne.
    drop_icon: discard detections confined to the bottom-right body-mark area before anything else."""
    out = []
    for r in records:
        d = dets[r["id"]]
        masks = np.unpackbits(d["masks"], axis=-1)[:, :, :common.EVAL_SIZE].astype(bool)
        if drop_icon and len(masks):
            ok = np.array([not icon_only(m) for m in masks])
            d = dict(d, conf=d["conf"][ok])
            masks = masks[ok]
        keep = d["conf"] > threshold if strict else d["conf"] >= threshold
        gt = np.unpackbits(d["gt"], axis=-1)[:, :common.EVAL_SIZE].astype(bool)
        pred = masks[keep].any(0) if keep.any() else np.zeros_like(gt)
        dice, iou = common.scores(pred, gt)
        out.append(dict(r, icon_only=icon_only(pred), presence_score=float(d["conf"].max()) if len(d["conf"]) else 0.0,
                        n_pred=int(keep.sum()), pred_pixels=int(pred.sum()),
                        intersection=int((pred & gt).sum()), dice=dice, iou=iou))
    return out


def save_dets(path: Path, records, dets):
    flat = {}
    for i, r in enumerate(records):
        d = dets[r["id"]]
        flat[f"{i}_conf"], flat[f"{i}_masks"], flat[f"{i}_gt"] = d["conf"], d["masks"], d["gt"]
        flat[f"{i}_boxes"] = d["boxes"]
    np.savez_compressed(path, **flat)
    path.with_suffix(".json").write_text(json.dumps(records))


def load_dets(path: Path):
    records = json.loads(Path(path).with_suffix(".json").read_text())
    with np.load(path) as z:
        dets = {r["id"]: dict(conf=z[f"{i}_conf"], boxes=z[f"{i}_boxes"], masks=z[f"{i}_masks"], gt=z[f"{i}_gt"])
                for i, r in enumerate(records)}
    return records, dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="busi:val,busi:test,uclm:all", help="comma list of source:split")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    results = {}
    for spec in args.sets.split(","):
        print("predict", spec, flush=True)
        recs, dets = predict_items(model, sources.items(spec), args.imgsz)
        save_dets(out / f"{spec.replace(':', '_')}_dets.npz", recs, dets)
        results[spec] = (recs, dets)
    val_spec = next(s for s in results if s.endswith(":val"))
    vrec, vdet = results[val_spec]
    calib = max([d["presence_score"] for d in dets_to_records(vrec, vdet, 1.0) if d["category"] == "normal"], default=0.0)
    report = dict(calibrated_threshold=calib, calibration_split=val_spec)
    for spec, (recs, dets) in results.items():
        report[spec] = dict(default_0p25=metrics(dets_to_records(recs, dets, 0.25, strict=False)),
                            calibrated=metrics(dets_to_records(recs, dets, calib)))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
