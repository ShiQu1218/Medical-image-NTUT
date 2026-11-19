"""MedSAM (Ma et al., Nat Commun 2024; weights wanglab/medsam-vit-base) with box prompts.

Preprocessing follows the official MedSAM_Inference.py, not the HF SamProcessor defaults:
image resized to 1024x1024 (bicubic), min-max normalised to [0,1], 3 channels; boxes scaled to 1024;
single-mask output; low-res logits -> sigmoid -> bilinear to original size -> > 0.5 (via to_eval_res).
Modes: gt_box (GT instance boxes) | yolo:<eval dir or npz>:<thr> (YOLO boxes with conf > thr).
Writes runs/medsam/<spec>_<tag>.jsonl in the wevis_eval.py schema.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import common
import sources


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def embed(model, gray):
    """1024x1024 bicubic resize + ImageNet normalisation (as in the HF preprocessor_config of these weights).
    Chosen on BUSI *val* (30 lesion images, GT boxes): ImageNet norm 0.906 vs official-repo [0,1] input 0.705."""
    img = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    x = torch.from_numpy(np.repeat(img[None], 3, 0))[None]
    return model.get_image_embeddings(pixel_values=((x - MEAN) / STD).cuda())


@torch.no_grad()
def segment(model, emb, boxes01, hw):
    h, w = hw
    probs = []
    for b in boxes01:
        box = torch.tensor([[[b[0] * 1024, b[1] * 1024, b[2] * 1024, b[3] * 1024]]], dtype=torch.float32).cuda()
        out = model(image_embeddings=emb, input_boxes=box, multimask_output=False)
        low = out.pred_masks[0, 0, :1]  # (1, 256, 256) logits
        p = torch.sigmoid(F.interpolate(low[None], size=(h, w), mode="bilinear", align_corners=False))[0, 0]
        probs.append(p.float().cpu().numpy())
    return np.max(np.stack(probs), 0) if probs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="busi:test,uclm:all")
    ap.add_argument("--modes", default="gt_box")
    ap.add_argument("--out", default=str(common.RUNS / "medsam"))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items only")
    ap.add_argument("--decoder", default=None, help="fine-tuned mask_decoder state dict (medsam_finetune.py)")
    args = ap.parse_args()
    from transformers import SamModel
    model = SamModel.from_pretrained("wanglab/medsam-vit-base").cuda().eval()
    if args.decoder:
        model.mask_decoder.load_state_dict(torch.load(args.decoder, map_location="cpu", weights_only=True), strict=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_args.json" if args.decoder else out / "args.json").write_text(
        json.dumps(dict(model="wanglab/medsam-vit-base", val_spec="busi:val", decoder=args.decoder), indent=2))
    for spec in args.sets.split(","):
        for mode in args.modes.split(","):
            yolo, tag = None, mode
            if mode.startswith("yolo:"):
                from yolo_eval import load_dets
                npz, thr = mode[len("yolo:"):].rsplit(":", 1)
                npz = Path(npz)
                npz = npz / f"{spec.replace(':', '_')}_dets.npz" if npz.is_dir() else npz
                yolo, thr = load_dets(npz)[1], float(thr)
                tag = f"yolo{thr:g}_{npz.parent.parent.name}"  # e.g. yolo0.475_busi_m_s42
            path = out / f"{spec.replace(':', '_')}_{tag}.jsonl"
            if path.exists():
                continue
            recs = []
            for k, it in enumerate(sources.items(spec)):
                if args.limit and k >= args.limit:
                    break
                gray = np.asarray(it["image"].convert("L"))
                extra = {}
                if yolo is None:
                    boxes = [common.tight_box(m) for m in it["instances"]]
                else:
                    det = yolo[it["id"]]
                    boxes = [list(map(float, b)) for b, c in zip(det["boxes"], det["conf"]) if c > thr]
                    extra = dict(presence_score=float(det["conf"].max()) if len(det["conf"]) else 0.0)
                prob = segment(model, embed(model, gray), boxes, gray.shape) if boxes else None
                pred, gt = common.to_eval_res(prob, it["union"])
                dice, iou = common.scores(pred, gt)
                recs.append(dict({x: v for x, v in it.items() if x not in ("image", "union", "instances")},
                                 mode=tag, **extra, n_pred=len(boxes), pred_pixels=int(pred.sum()),
                                 intersection=int((pred.astype(bool) & gt.astype(bool)).sum()), dice=dice, iou=iou,
                                 pred_bits=np.packbits(pred.astype(bool)).tobytes().hex()))
                if (k + 1) % 100 == 0:
                    print(f"  {spec} {tag} {k + 1}", flush=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
            les = [r["dice"] for r in recs if r["category"] != "normal"]
            print(spec, tag, "lesion dice", round(float(np.mean(les)), 4), flush=True)


if __name__ == "__main__":
    main()
