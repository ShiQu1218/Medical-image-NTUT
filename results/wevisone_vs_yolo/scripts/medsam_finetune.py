"""Fine-tune MedSAM's mask decoder on BUSI train (same 368 clean images as WeVisOne/YOLO/nnU-Net).

Fair counterpart to the WeVisOne pixel-decoder fine-tune: image encoder + prompt encoder frozen, mask decoder
(4.1M params) trained with GT instance boxes jittered ±5% (as train.py), BCE + Dice on the 256x256 low-res
logits, horizontal flip (embeddings precomputed for both orientations). AdamW 1e-4, cosine, 40 epochs,
batch 8 instances. Checkpoint chosen on busi:val lesion Dice with GT boxes (eval at 256 like everything else).
Writes runs/medsam_ft/{best_decoder.pt, history.csv, args.json}. Evaluate with
  medsam_eval.py --decoder runs/medsam_ft/best_decoder.pt --out runs/medsam_ft ...
"""
from __future__ import annotations

import argparse
import csv
import json
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import common
import sources
from medsam_eval import embed, segment

OUT = common.RUNS / "medsam_ft"


def jitter(box, r=0.05):
    w, h = box[2] - box[0], box[3] - box[1]
    out = [box[0] + random.uniform(-r, r) * w, box[1] + random.uniform(-r, r) * h,
           box[2] + random.uniform(-r, r) * w, box[3] + random.uniform(-r, r) * h]
    out = [min(1.0, max(0.0, v)) for v in out]
    return out if out[2] - out[0] > 1e-3 and out[3] - out[1] > 1e-3 else box


@torch.no_grad()
def precompute(model, spec):
    data = []
    for it in sources.items(spec):
        gray = np.asarray(it["image"].convert("L"))
        for flip in (False, True):
            g = np.ascontiguousarray(gray[:, ::-1]) if flip else gray
            inst = [np.ascontiguousarray(m[:, ::-1]) if flip else m for m in it["instances"]]
            if not inst:
                continue
            emb = embed(model, g).half().cpu()
            tg = [torch.from_numpy(cv2.resize(m.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)) for m in inst]
            data.append(dict(emb=emb, boxes=[common.tight_box(m) for m in inst], targets=tg, flip=flip))
    return data


def val_dice(model, items):
    model.eval()
    d = []
    for it in items:
        gray = np.asarray(it["image"].convert("L"))
        boxes = [common.tight_box(m) for m in it["instances"]]
        prob = segment(model, embed(model, gray), boxes, gray.shape)
        pred, gt = common.to_eval_res(prob, it["union"])
        d.append(common.scores(pred, gt)[0])
    return float(np.mean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "args.json").write_text(json.dumps(dict(vars(args), model="wanglab/medsam-vit-base", val_spec="busi:val",
                                                   trained="mask_decoder only"), indent=2))
    from transformers import SamModel
    model = SamModel.from_pretrained("wanglab/medsam-vit-base").cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.mask_decoder.parameters():
        p.requires_grad_(True)
    train = precompute(model.eval(), "busi:train")
    val_items = [it for it in sources.items("busi:val") if it["instances"]]
    pairs = [(k, j) for k, d in enumerate(train) for j in range(len(d["boxes"]))]
    steps = args.epochs * ((len(pairs) + args.batch - 1) // args.batch)
    opt = torch.optim.AdamW(model.mask_decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, hist = val_dice(model, val_items), []
    print("zero-shot val lesion dice", round(best, 4), flush=True)
    torch.save(model.mask_decoder.state_dict(), OUT / "best_decoder.pt")
    hist.append(dict(epoch=0, loss=None, val_lesion_dice=best))
    for ep in range(1, args.epochs + 1):
        model.mask_decoder.train()
        random.shuffle(pairs)
        losses = []
        for b in range(0, len(pairs), args.batch):
            chunk = pairs[b:b + args.batch]
            emb = torch.cat([train[k]["emb"] for k, _ in chunk]).cuda().float()
            box = torch.tensor([[[v * 1024 for v in jitter(train[k]["boxes"][j])]] for k, j in chunk], dtype=torch.float32).cuda()
            tgt = torch.stack([train[k]["targets"][j] for k, j in chunk]).cuda()
            out = model(image_embeddings=emb, input_boxes=box, multimask_output=False)
            logit = out.pred_masks[:, 0, 0]
            prob = torch.sigmoid(logit)
            dice = 1 - (2 * (prob * tgt).sum((1, 2)) + 1) / (prob.sum((1, 2)) + tgt.sum((1, 2)) + 1)
            loss = F.binary_cross_entropy_with_logits(logit, tgt) + dice.mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            losses.append(loss.item())
        v = val_dice(model, val_items)
        hist.append(dict(epoch=ep, loss=float(np.mean(losses)), val_lesion_dice=v))
        print(f"epoch {ep} loss {np.mean(losses):.4f} val {v:.4f}", flush=True)
        if v > best:
            best = v
            torch.save(model.mask_decoder.state_dict(), OUT / "best_decoder.pt")
    with (OUT / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
    (OUT / "best.json").write_text(json.dumps(max(hist, key=lambda h: h["val_lesion_dice"]), indent=2))
    (OUT / "TRAIN_DONE").write_text("")


if __name__ == "__main__":
    main()
