"""Convert nnUNetv2_predict outputs (--save_probabilities) into our record format.

Writes runs/nnunet/<name>/<spec key>_generate.jsonl (same schema as wevis_eval.py) so analyze.py's
"wevis" loader works unchanged. presence_score = max foreground probability in the image; masks use
the foreground probability map through common.to_eval_res (bilinear to 256, > 0.5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import common
import sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="nnUNetv2_predict output folder")
    ap.add_argument("--images", required=True, help="imagesTs_* folder holding cases.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_spec", required=True, help="written to args.json for gate calibration")
    ap.add_argument("--delete_probs", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(dict(val_spec=args.val_spec, model="nnU-Net v2 2d"), indent=2))
    meta = json.loads((Path(args.images) / "cases.json").read_text())
    by_id = {it_id: cid for cid, it_id in meta["cases"].items()}
    path = out / f"{meta['spec'].replace(':', '_')}_generate.jsonl"
    tmp = path.with_suffix(".partial")  # renamed only on success, so the queue never skips a half-written file
    with tmp.open("w", encoding="utf-8") as stream:
        for it in sources.items(meta["spec"]):
            cid = by_id[it["id"]]
            with np.load(Path(args.pred) / f"{cid}.npz") as z:
                prob = z["probabilities"][1].astype(np.float32).squeeze()
            if prob.shape != it["union"].shape:
                # e.g. one BUS_UC image is 256x256 with a 256x257 mask; both go to 256x256 below anyway
                print(f"warning: shape {it['id']} prob {prob.shape} vs gt {it['union'].shape}")
            pred, gt = common.to_eval_res(prob, it["union"])
            dice, iou = common.scores(pred, gt)
            n = cv2.connectedComponents(pred.astype(np.uint8))[0] - 1
            rec = dict({k: v for k, v in it.items() if k not in ("image", "union", "instances")},
                       mode="generate", presence_score=float(prob.max()), n_pred=int(n),
                       pred_pixels=int(pred.sum()), intersection=int((pred.astype(bool) & gt.astype(bool)).sum()),
                       dice=dice, iou=iou, pred_bits=np.packbits(pred.astype(bool)).tobytes().hex())
            stream.write(json.dumps(rec) + "\n")
            if args.delete_probs:
                (Path(args.pred) / f"{cid}.npz").unlink()
                (Path(args.pred) / f"{cid}.pkl").unlink(missing_ok=True)
    tmp.replace(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
