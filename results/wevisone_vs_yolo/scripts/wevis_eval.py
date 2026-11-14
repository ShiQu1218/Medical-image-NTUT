"""WeVisOne-VL evaluation over any sources spec, with resumable per-image journals.

Modes
  generate          : presence score + model-generated boxes/masks (end to end)
  gt_box            : GT boxes forced (oracle; isolates the pixel decoder)
  yolo:<npz>:<thr>  : hybrid; boxes from YOLO detections with conf > thr (yolo_eval.py output)
Records match fpr_metrics.metrics(); presence_score is always the WeVisOne logit gap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common
import sources
from evaluate import apply_finetune, predict_union
from fpr_metrics import metrics
from presence_experiment import presence_score


def run(engine, spec: str, mode: str, journal: Path, yolo=None):
    done = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
    with journal.open("a", encoding="utf-8") as stream:
        for k, it in enumerate(sources.items(spec)):
            if it["id"] in done:
                continue
            image, union = it["image"], it["union"]
            extra = {}
            if mode == "generate":
                extra = presence_score(engine, image)
                response, prob, n = predict_union(engine, image, it["instances"], "generate")
            elif mode == "gt_box":
                response, prob, n = predict_union(engine, image, it["instances"], "gt_box")
            else:
                d, thr = yolo
                det = d[it["id"]]
                boxes = [[float(v) for v in b] for b, c in zip(det["boxes"], det["conf"]) if c > thr]
                extra = dict(presence_score=float(det["conf"].max()) if len(det["conf"]) else 0.0)
                if boxes:
                    response = common.target_response(boxes)
                    _, masks = engine.decode_masks(image, common.PROMPT, response, boxes=boxes)
                    prob, n = np.max(np.stack(masks), axis=0), len(masks)
                else:
                    response, prob, n = "None", None, 0
            pred, gt = common.to_eval_res(prob, union)
            dice, iou = common.scores(pred, gt)
            rec = dict({x: v for x, v in it.items() if x not in ("image", "union", "instances")},
                       mode=mode, **extra, response=response, n_pred=n, pred_pixels=int(pred.sum()),
                       intersection=int((pred.astype(bool) & gt.astype(bool)).sum()), dice=dice, iou=iou,
                       pred_bits=np.packbits(pred.astype(bool)).tobytes().hex())
            stream.write(json.dumps(rec) + "\n")
            stream.flush()
            done[it["id"]] = rec
            if (k + 1) % 50 == 0:
                print(f"  {spec} {mode} {k + 1}", flush=True)
    return list(done.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora")
    ap.add_argument("--decoder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="busi:val,busi:test,uclm:all")
    ap.add_argument("--modes", default="generate,gt_box")
    ap.add_argument("--prompt_suffix", default="", help="text appended (after a newline) to the training prompt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.prompt_suffix:
        common.PROMPT = common.PROMPT + "\n" + args.prompt_suffix  # read at call time by every inference path
    (out / "prompt.txt").write_text(common.PROMPT, encoding="utf-8")
    engine = common.load_engine()
    apply_finetune(engine, args.lora, args.decoder)
    report = {}
    for spec in args.sets.split(","):
        for mode in args.modes.split(","):
            yolo, tag = None, mode
            if mode.startswith("yolo:"):
                from yolo_eval import load_dets
                npz, thr = mode[len("yolo:"):].rsplit(":", 1)  # npz file or the eval dir holding <spec>_dets.npz
                npz = Path(npz)
                npz = npz.parent / f"{spec.replace(':', '_')}_dets.npz" if npz.is_dir() else npz
                yolo = (load_dets(npz)[1], float(thr))
                tag = f"yolo{float(thr):g}"
            recs = run(engine, spec, mode if yolo is None else "yolo", out / f"{spec.replace(':', '_')}_{tag}.jsonl", yolo)
            report[f"{spec}/{tag}"] = metrics(recs)
            print(spec, tag, json.dumps(report[f"{spec}/{tag}"]), flush=True)
    path = out / "report.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **report}, indent=2))


if __name__ == "__main__":
    main()
