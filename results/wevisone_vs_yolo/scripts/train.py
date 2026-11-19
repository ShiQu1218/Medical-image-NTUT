"""Joint fine-tuning on cleaned BUSI: LoRA on the Qwen3-VL LLM + full pixel decoder.

One teacher-forced forward per image gives
  * LM cross-entropy on the assistant answer (boxes + <OBJ>, or "None")   -> LoRA
  * <OBJ> hidden state + ViT pre-merge features -> pixel decoder -> BCE+Dice -> decoder (+ LoRA)
The ViT stays frozen. Model selection: best end-to-end Dice on the 85 clean val images.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoKrConfig, LoraConfig, get_peft_model
from PIL import Image

import common
import sources
from fpr_metrics import hard_negative_ids
from evaluate import predict_union, summarize
from wevisone_vl.engine import _pil_to_float_rgb
from wevisone_vl.feature_bridge import (
    LastHiddenCapture,
    VisualFeatureCapture,
    extract_t_seg,
    per_sample_premerge_feature_maps,
    premerge_grid_hw,
)

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def mask_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + soft Dice; target resized (area) to the logit resolution."""
    t = F.interpolate(target[:, None], size=logits.shape[-2:], mode="area")[:, 0]
    bce = F.binary_cross_entropy_with_logits(logits, t)
    p = logits.sigmoid().flatten(1)
    t = t.flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
    return bce + dice.mean()


def jitter(box: list[float], scale: float) -> list[float]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    d = [random.uniform(-scale, scale) for _ in range(4)]
    out = [x1 + d[0] * w, y1 + d[1] * h, x2 + d[2] * w, y2 + d[3] * h]
    out = [min(1.0, max(0.0, v)) for v in out]
    if out[2] - out[0] < 1e-3 or out[3] - out[1] < 1e-3:
        return box
    return out


def strong_aug(image: Image.Image, instances: list[np.ndarray]):
    """Ultrasound-style photometric + mild geometric augmentation (applied after the flip)."""
    g = np.asarray(image.convert("L")).astype(np.float32) / 255.0
    h, w = g.shape
    if random.random() < 0.7:  # scale / translate / rotate
        s = random.uniform(0.8, 1.2)
        a = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, s)
        M[:, 2] += [random.uniform(-0.1, 0.1) * w, random.uniform(-0.1, 0.1) * h]
        g = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        instances = [cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) for m in instances]
        instances = [m for m in instances if m.sum() >= 16]  # lesion moved out of frame -> dropped
    if random.random() < 0.8:  # gamma, contrast, brightness (scanner gain / TGC differences)
        g = np.clip(g, 0, 1) ** random.uniform(0.6, 1.6)
        g = (g - g.mean()) * random.uniform(0.75, 1.3) + g.mean() + random.uniform(-0.1, 0.1)
    if random.random() < 0.3:  # multiplicative speckle, gamma noise with mean 1 (looks = L)
        L = random.uniform(8, 30)
        g = g * np.random.gamma(L, 1.0 / L, g.shape).astype(np.float32)
    if random.random() < 0.25:
        k = random.choice([3, 5])
        g = cv2.GaussianBlur(g, (k, k), 0)
    if random.random() < 0.2:  # resolution loss
        f = random.uniform(0.4, 0.8)
        g = cv2.resize(cv2.resize(g, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA), (w, h))
    g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(g).convert("RGB"), instances


def step_sample(engine, model, decoder, item, args):
    image, instances = item["image"], list(item["instances"])
    if random.random() < 0.5:  # horizontal flip
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        instances = [np.ascontiguousarray(m[:, ::-1]) for m in instances]
    if args.aug == "strong":
        image, instances = strong_aug(image, instances)
    boxes = [common.tight_box(m) for m in instances]
    response = common.target_response(boxes)

    prefix = engine._inputs(engine._messages(image, common.PROMPT), image, generation=True)
    messages = engine._messages(image, common.PROMPT) + [{"role": "assistant", "content": response}]
    inputs = engine._inputs(messages, image, generation=False)
    n_prefix = prefix["input_ids"].shape[1]
    assert torch.equal(inputs["input_ids"][0, :n_prefix], prefix["input_ids"][0]), "prefix mismatch"
    labels = inputs["input_ids"].clone()
    labels[:, :n_prefix] = -100

    with VisualFeatureCapture(model, feature_layers=engine.cfg.vision_feature_layers) as vc, \
            LastHiddenCapture(model) as hc:
        out = model(**inputs, labels=labels, use_cache=False, return_dict=True)
    lm_loss = out.loss
    seg_loss = torch.zeros((), device=lm_loss.device)

    negative_box = not boxes and args.negative_boxes
    if negative_box:
        # A second response supplies an OBJ state while the normal LM target stays None.
        x1, y1 = random.uniform(0.05, 0.55), random.uniform(0.05, 0.55)
        boxes = [[x1, y1, min(0.98, x1 + random.uniform(0.15, 0.4)),
                  min(0.98, y1 + random.uniform(0.15, 0.4))]]
        response = common.target_response(boxes)
        messages = engine._messages(image, common.PROMPT) + [{"role": "assistant", "content": response}]
        inputs = engine._inputs(messages, image, generation=False)
        with VisualFeatureCapture(model, feature_layers=engine.cfg.vision_feature_layers) as vc, LastHiddenCapture(model) as hc:
            model(**inputs, use_cache=False, return_dict=True, logits_to_keep=1)

    if boxes:
        t_seg = extract_t_seg(hc.hidden, inputs["input_ids"], engine.seg_token_id)[0]
        grid_thw = inputs["image_grid_thw"]
        visual = per_sample_premerge_feature_maps(
            vc.premerge_features, engine.cfg.vision_feature_layers, grid_thw, engine.spatial_merge_size
        )[0]
        visual = [v.detach() for v in visual]  # ViT frozen
        size = engine.cnn_size
        cnn = _pil_to_float_rgb(image.resize((size, size), Image.Resampling.BICUBIC))[None].to(engine.device)
        boxes_t = torch.tensor([jitter(b, args.box_jitter) for b in boxes], device=engine.device)
        dec = decoder(image=cnn, visual_features_2d=visual, t_seg=t_seg, boxes01=boxes_t,
                      grid_hw=premerge_grid_hw(grid_thw)[0])
        targets = [np.zeros((image.height, image.width), dtype=np.uint8)] if negative_box else instances
        gt = torch.from_numpy(np.stack([
            cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) for m in targets
        ])).float().to(engine.device)
        seg_loss = (mask_loss(dec["mask_logits"], gt)
                    + 0.5 * mask_loss(dec["mask_logits_aux"], gt)
                    + 0.25 * mask_loss(dec["mask_logits_coarse"], gt))
    return lm_loss, seg_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.RUNS / "lora_decoder"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lr_decoder", type=float, default=3e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--adapter", choices=["lora", "lokr"], default="lora",
                    help="lokr: PEFT LoKr (Kronecker) on the same target modules; --rank = LoKr r, alpha = r; lr from --lr_lora")
    ap.add_argument("--lokr_factor", type=int, default=-1, help="LoKr decompose_factor (-1 = PEFT default)")
    ap.add_argument("--lokr_decompose_both", action="store_true")
    ap.add_argument("--optimizer", choices=["adamw", "automagic3"], default="adamw",
                    help="automagic3: ostris ai-toolkit Automagic3 (third_party/), fused=False for grad accumulation; "
                         "--lr_lora/--lr_decoder are the START lrs of its two param groups, adapted by the optimizer")
    ap.add_argument("--schedule", choices=["cosine", "constant"], default="cosine",
                    help="cosine: 5%% warmup + cosine to 0 (default). constant: no external schedule (required for automagic3)")
    ap.add_argument("--seg_weight", type=float, default=1.0)
    ap.add_argument("--box_jitter", type=float, default=0.05)
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after N optimizer steps")
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normal_repeat", type=int, default=1, help="copies of each normal image in the epoch pool")
    ap.add_argument("--hard_negatives", help="Training-only manifest from presence_experiment.py")
    ap.add_argument("--hard_negative_weight", type=float, default=2.0, help="LM loss multiplier for mined normal images; preserves lesion exposure")
    ap.add_argument("--save_each_epoch", action="store_true")
    ap.add_argument("--init_lora")
    ap.add_argument("--init_decoder")
    ap.add_argument("--negative_boxes", action="store_true", help="Supervise random normal-image boxes with empty masks")
    ap.add_argument("--train_spec", default="busi:train", help="sources.py spec")
    ap.add_argument("--val_spec", default="busi:val", help="sources.py spec used for checkpoint selection")
    ap.add_argument("--aug", choices=["flip", "strong"], default="flip")
    ap.add_argument("--total_steps", type=int, default=0,
                    help="fixed optimizer-update budget; overrides --epochs (epochs = ceil(total/steps_per_epoch))")
    ap.add_argument("--n_vals", type=int, default=0, help="validate ~this many times, evenly spaced (0 = every epoch)")
    args = ap.parse_args()
    if args.normal_repeat < 1 or not math.isfinite(args.hard_negative_weight) or args.hard_negative_weight < 1:
        ap.error("normal_repeat and hard_negative_weight must be >= 1 and finite")
    rows = sources.rows(args.train_spec)
    val_rows = sources.rows(args.val_spec)
    if args.total_steps:
        args.epochs = math.ceil(args.total_steps / math.ceil(len(rows) / args.accum))
    val_epochs = set(range(1, args.epochs + 1)) if not args.n_vals else \
        {max(1, round(args.epochs * (i + 1) / args.n_vals)) for i in range(args.n_vals)}
    hard_ids = hard_negative_ids(json.loads(Path(args.hard_negatives).read_text()), rows) if args.hard_negatives else set()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    engine = common.load_engine()
    model = engine.model
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    if args.init_lora:
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(model, args.init_lora, is_trainable=True)
    else:
        cfg = (LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=LORA_TARGETS)
               if args.adapter == "lora" else
               LoKrConfig(r=args.rank, alpha=args.rank, decompose_factor=args.lokr_factor,
                          decompose_both=args.lokr_decompose_both, target_modules=LORA_TARGETS))
        peft_model = get_peft_model(model, cfg)
    if args.init_decoder:
        engine.mask_decoder.load_state_dict(torch.load(args.init_decoder, map_location='cpu', weights_only=True))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n or "lokr_" in n]  # adapter params
    for p in lora_params:
        p.requires_grad_(True)
    decoder = engine.mask_decoder.float()
    for p in decoder.parameters():
        p.requires_grad_(True)
    print(f"LoRA params: {sum(p.numel() for p in lora_params):,}  decoder params: "
          f"{sum(p.numel() for p in decoder.parameters()):,}", flush=True)

    steps_per_epoch = math.ceil(len(rows) / args.accum)
    total = args.max_steps or args.total_steps or steps_per_epoch * args.epochs
    print(f"train {args.train_spec}: {len(rows)} imgs, val {args.val_spec}: {len(val_rows)}, "
          f"{args.epochs} epochs, {total} updates, val at {sorted(val_epochs)}", flush=True)
    groups = [{"params": lora_params, "lr": args.lr_lora}, {"params": list(decoder.parameters()), "lr": args.lr_decoder}]
    if args.optimizer == "automagic3":
        if args.schedule != "constant":
            ap.error("automagic3 adapts its own lr; use --schedule constant")
        from third_party.automagic3 import Automagic3
        opt = Automagic3(groups, fused=False)
    else:
        opt = torch.optim.AdamW(groups, weight_decay=0.0)
    warmup = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, (lambda s: 1.0) if args.schedule == "constant" else
        (lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))

    def current_lrs():  # lr actually applied at the last update (Automagic3 keeps its adapted lr in state)
        if hasattr(opt, "get_learning_rates"):
            return opt.get_learning_rates()
        return [g["lr"] for g in opt.param_groups]

    log = (out / "history.csv").open("a", newline="")
    writer = csv.writer(log)
    writer.writerow(["epoch", "step", "lm_loss", "seg_loss", "val_dice", "val_lesion_dice", "val_normal_fp", "seconds",
                     "lr_adapter", "lr_decoder"])
    best, step, started = -1.0, 0, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        decoder.train()
        # Oversample normals while keeping the per-epoch sample count (= update budget) fixed.
        pool = rows + [r for r in rows if r["category"] == "normal"] * (args.normal_repeat - 1)
        random.shuffle(pool)
        epoch_rows = pool[:len(rows)]
        n_norm = sum(r["category"] == "normal" for r in epoch_rows)
        print(f"epoch {epoch}: {len(epoch_rows)} samples, {n_norm} normal", flush=True)
        lm_acc, seg_acc, n_acc = 0.0, 0.0, 0
        for i, row in enumerate(epoch_rows):
            lm_loss, seg_loss = step_sample(engine, model, decoder, sources.load(row), args)
            if hard_ids and row.get("image") in hard_ids:  # non-BUSI rows (e.g. BUS-BRA) carry no 'image' key
                lm_loss = lm_loss * args.hard_negative_weight
            ((lm_loss + args.seg_weight * seg_loss) / args.accum).backward()
            lm_acc, seg_acc, n_acc = lm_acc + lm_loss.item(), seg_acc + seg_loss.item(), n_acc + 1
            if (i + 1) % args.accum == 0 or i + 1 == len(epoch_rows):
                if args.optimizer == "adamw":  # Automagic3 has its own trust region (update RMS/element clip)
                    torch.nn.utils.clip_grad_norm_(lora_params + list(decoder.parameters()), 1.0)
                lr_used = current_lrs()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0:
                    print(f"ep{epoch} step{step}/{total} lm={lm_acc / n_acc:.4f} seg={seg_acc / n_acc:.4f} "
                          f"{time.perf_counter() - started:.0f}s", flush=True)
                    writer.writerow([epoch, step, lm_acc / n_acc, seg_acc / n_acc, "", "", "", round(time.perf_counter() - started),
                                     lr_used[0], lr_used[1]])
                    log.flush()
                    lm_acc, seg_acc, n_acc = 0.0, 0.0, 0
                if (args.max_steps and step >= args.max_steps) or step >= total:
                    break
        if epoch not in val_epochs and step < total:
            continue
        model.eval()
        decoder.eval()
        summary = validate(engine, val_rows[:args.val_limit] if args.val_limit else val_rows)
        print(f"epoch {epoch} VAL {json.dumps(summary)}", flush=True)
        writer.writerow([epoch, step, "", "", summary["dice"], summary["lesion_dice"], summary["normal_fp_rate"],
                         round(time.perf_counter() - started)])
        log.flush()
        if args.save_each_epoch:
            epoch_out = out / f"epoch_{epoch}"
            peft_model.save_pretrained(epoch_out / "lora")
            torch.save(decoder.state_dict(), epoch_out / "decoder.pt")
            (epoch_out / "metrics.json").write_text(json.dumps(summary, indent=2))
        if summary["dice"] > best:
            best = summary["dice"]
            peft_model.save_pretrained(out / "best_lora")  # adapter only
            torch.save(decoder.state_dict(), out / "best_decoder.pt")
            (out / "best.json").write_text(json.dumps(dict(epoch=epoch, step=step, **summary), indent=2))
            print(f"  saved best (val dice {best:.4f})", flush=True)
        if (args.max_steps and step >= args.max_steps) or step >= total:
            break
    log.close()


@torch.no_grad()
def validate(engine, rows):
    per_image = []
    for row in rows:
        it = sources.load(row)
        response, prob, n = predict_union(engine, it["image"], it["instances"], "generate")
        pred, gt = common.to_eval_res(prob, it["union"])
        dice, iou = common.scores(pred, gt)
        native = (prob > 0.5).astype(np.uint8) if prob is not None else np.zeros_like(it["union"])
        per_image.append(dict(category=it["category"], dice=dice, iou=iou, native_dice=common.scores(native, it["union"])[0],
                              n_pred=n, pred_pixels=int(pred.sum()),
                              intersection=int((pred.astype(bool) & gt.astype(bool)).sum())))
    return dict(summarize(per_image), n=len(rows))


if __name__ == "__main__":
    main()
