"""Inference cost on one RTX 4090: params, per-image latency (batch 1), peak VRAM.

Run one model per process so peak VRAM is clean:
  python bench_cost.py --model wevis|yolo|nnunet [--n 60]
Writes runs/comparison/cost_<model>.json. Images: first n of busi:test (native resolution, PIL RGB),
5 warm-up images excluded. Latency covers the full inference path used in the reports:
  wevis : presence forward (gate) + greedy generate + mask decode
  yolo  : model.predict(imgsz=640, retina_masks) incl. pre/post-processing
  nnunet: nnUNetPredictor.predict_single_npy_array with default mirroring TTA (as nnUNetv2_predict)
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import common
import sources

RUN = common.RUNS


def n_params(*modules):
    seen, total = set(), 0
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def timed(fn, items, warm=5):
    lat = []
    for i, it in enumerate(items):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn(it)
        torch.cuda.synchronize()
        if i >= warm:
            lat.append(time.perf_counter() - t)
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["wevis", "yolo", "nnunet"])
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    items = list(sources.items("busi:test"))[:args.n + 5]
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    extra = {}
    if args.model == "wevis":
        from evaluate import apply_finetune, predict_union
        from presence_experiment import presence_score
        eng = common.load_engine()
        r = RUN / "wevis" / "baseline_s42"
        apply_finetune(eng, str(r / "best_lora"), str(r / "best_decoder.pt"))
        params = n_params(eng.model, eng.mask_decoder)
        extra["trainable_finetune_params"] = "LoRA r16 33M + decoder 9.6M (merged at inference)"

        def fn(it):
            presence_score(eng, it["image"])
            predict_union(eng, it["image"], it["instances"], "generate")
    elif args.model == "yolo":
        import cv2
        from ultralytics import YOLO
        m = YOLO(str(RUN / "yolo" / "busi_m_s42" / "weights" / "best.pt"))
        params = n_params(m.model)

        def fn(it):
            bgr = cv2.cvtColor(np.asarray(it["image"]), cv2.COLOR_RGB2BGR)
            m.predict(bgr, conf=0.001, iou=0.7, max_det=20, imgsz=640, retina_masks=True, verbose=False, device=0)
    else:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        p = nnUNetPredictor(tile_step_size=0.5, use_mirroring=True, device=torch.device("cuda", 0), verbose=False,
                            verbose_preprocessing=False, allow_tqdm=False)
        p.initialize_from_trained_model_folder(
            str(common.ROOT / "nnunet/results/Dataset501_BUSIclean/nnUNetTrainer_250epochs__nnUNetPlans__2d"),
            use_folds=(0,), checkpoint_name="checkpoint_final.pth")
        params = n_params(p.network)

        def fn(it):
            g = np.asarray(it["image"].convert("L"), np.float32)[None, None]  # (c, z=1, y, x)
            p.predict_single_npy_array(g, {"spacing": [999, 1, 1]}, None, None, True)
    lat = np.array(timed(fn, items)) * 1000
    out = dict(model=args.model, params=params, n_images=len(lat), latency_ms_mean=float(lat.mean()),
               latency_ms_median=float(np.median(lat)), latency_ms_p95=float(np.percentile(lat, 95)),
               images_per_s=float(1000 / lat.mean()),
               peak_vram_gb=float((torch.cuda.max_memory_allocated() - base) / 2 ** 30),
               gpu=torch.cuda.get_device_name(0), **extra)
    (RUN / "comparison" / f"cost_{args.model}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
