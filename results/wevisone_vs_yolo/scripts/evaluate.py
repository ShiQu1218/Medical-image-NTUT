"""Evaluate WeVisOne-VL on the cleaned BUSI split with the U-Net baseline's metrics.

mode=generate : model predicts boxes + <OBJ> itself (end-to-end, the real test)
mode=gt_box   : boxes forced from GT (oracle; isolates pixel-decoder quality)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import common


def apply_finetune(engine, lora: str | None, decoder: str | None, vision: str | None = None):
    if lora:
        from peft import PeftModel

        engine.model = PeftModel.from_pretrained(engine.model, lora).merge_and_unload().eval()
    if vision:  # train.py --vit_full: whole vision tower state dict
        names = [n for n, _ in engine.model.named_modules() if n.split(".")[-1] == "visual"]
        mod = engine.model.get_submodule(min(names, key=len))
        mod.load_state_dict(torch.load(vision, map_location="cpu", weights_only=True), strict=True)
    if decoder:
        state = torch.load(decoder, map_location="cpu", weights_only=True)
        engine.mask_decoder.load_state_dict(state, strict=True)
        engine.mask_decoder.eval()


def predict_union(engine, image, instances, mode: str):
    """Return (response, union probability map or None, n_boxes)."""
    if mode == "gt_box":
        if not instances:
            return "None", None, 0
        boxes = [common.tight_box(m) for m in instances]
        response = common.target_response(boxes)
        _, masks = engine.decode_masks(image, common.PROMPT, response, boxes=boxes)
    else:
        response = engine.generate_text(image, common.PROMPT, max_new_tokens=128,
                                        temperature=0.0, repetition_penalty=1.0)
        try:
            _, masks = engine.decode_masks(image, common.PROMPT, response)
        except RuntimeError:  # malformed response (e.g. box without <OBJ>)
            masks = []
    if not masks:
        return response, None, 0
    return response, np.max(np.stack(masks), axis=0), len(masks)


def evaluate(engine, split: str, mode: str, out: Path | None = None, limit: int = 0) -> dict:
    rows = common.clean_split(split)
    if limit:
        rows = rows[:limit]
    per_image = []
    preds = {}
    started = time.perf_counter()
    for row in rows:
        image, instances, union = common.load_sample(row)
        response, prob, n = predict_union(engine, image, instances, mode)
        pred256, gt256 = common.to_eval_res(prob, union)
        dice, iou = common.scores(pred256, gt256)
        native = (prob > 0.5).astype(np.uint8) if prob is not None else np.zeros_like(union)
        ndice, _ = common.scores(native, union)
        per_image.append(dict(image=row["image"], category=row["category"], dice=dice, iou=iou,
                              pred_pixels=int(pred256.sum()),
                              intersection=int((pred256.astype(bool) & gt256.astype(bool)).sum()),
                              native_dice=ndice, n_pred=n, n_gt=len(instances), response=response))
        preds[row["image"]] = pred256
    summary = summarize(per_image)
    summary.update(split=split, mode=mode, n=len(rows), seconds=round(time.perf_counter() - started, 1))
    if out:
        out.mkdir(parents=True, exist_ok=True)
        with (out / f"{split}_{mode}_per_image.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_image[0]))
            w.writeheader()
            w.writerows(per_image)
        (out / f"{split}_{mode}_summary.json").write_text(json.dumps(summary, indent=2))
        np.savez_compressed(out / f"{split}_{mode}_pred256.npz", **{k.replace("/", "__"): v for k, v in preds.items()})
    return summary


def summarize(per_image: list[dict]) -> dict:
    lesion = [r for r in per_image if r["category"] != "normal"]
    normal = [r for r in per_image if r["category"] == "normal"]
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    return dict(
        dice=mean([r["dice"] for r in per_image]),
        iou=mean([r["iou"] for r in per_image]),
        lesion_dice=mean([r["dice"] for r in lesion]),
        native_lesion_dice=mean([r["native_dice"] for r in lesion]),
        normal_fp_rate=mean([float(r["n_pred"] > 0 and r["dice"] < 1) for r in normal]),
        lesion_miss_rate=mean([float(r["n_pred"] == 0) for r in lesion]),
        lesion_empty_rate=mean([float(r["pred_pixels"] == 0) for r in lesion]),
        lesion_zero_overlap_rate=mean([float(r["intersection"] == 0) for r in lesion]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora")
    ap.add_argument("--decoder")
    ap.add_argument("--split", default="test")
    ap.add_argument("--modes", default="generate,gt_box")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    engine = common.load_engine()
    apply_finetune(engine, args.lora, args.decoder)
    for mode in args.modes.split(","):
        s = evaluate(engine, args.split, mode, Path(args.out), args.limit)
        print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
