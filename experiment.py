"""Paired original/cleaned BUSI experiments, with one common clean holdout."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset import load_records, load_arrays
from model import UNet
from preprocessing import METHODS, preprocess


def scores(prediction, target):
    prediction, target = prediction.bool().flatten(1), target.bool().flatten(1)
    intersection = (prediction & target).sum(1).float()
    total = prediction.sum(1) + target.sum(1)
    union = (prediction | target).sum(1)
    dice = torch.where(total == 0, 1.0, 2*intersection / total.clamp_min(1))
    iou = torch.where(union == 0, 1.0, intersection / union.clamp_min(1))
    return dice, iou


def loss_fn(logits, target):
    probability = logits.sigmoid()
    dice = (2*(probability*target).sum((1, 2, 3)) + 1) / (
        probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - dice).mean()


def augment(images, masks):
    angles = (torch.rand(len(images), device=images.device)*20 - 10)*np.pi/180
    scales = 0.95 + torch.rand(len(images), device=images.device)*0.1
    theta = torch.zeros(len(images), 2, 3, device=images.device)
    theta[:, 0, 0] = theta[:, 1, 1] = angles.cos()*scales
    theta[:, 0, 1], theta[:, 1, 0] = -angles.sin()*scales, angles.sin()*scales
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return (F.grid_sample(images, grid, align_corners=False),
            F.grid_sample(masks, grid, mode="nearest", align_corners=False))


@torch.inference_mode()
def predict(model, images, masks, indices, batch_size):
    model.eval()
    predictions, dices, ious = [], [], []
    for batch in indices.split(batch_size):
        output = model(images[batch]).sigmoid() >= 0.5
        dice, iou = scores(output, masks[batch] >= 0.5)
        predictions.append(output[:, 0].cpu().numpy())
        dices.extend(dice.cpu().tolist())
        ious.extend(iou.cpu().tolist())
    return np.concatenate(predictions), np.array(dices), np.array(ious)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(images, masks, train_indices, val_indices, args, seed, folder):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = UNet().to(images.device, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best, history = -1.0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), device=images.device)]
        loss_sum = 0.0
        for batch in order.split(args.batch_size):
            x, y = augment(images[batch], masks[batch])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(batch)
        _, dice, _ = predict(model, images, masks, val_indices, args.batch_size)
        value = float(dice.mean())
        history.append(dict(epoch=epoch, train_loss=loss_sum/len(order), val_dice=value))
        if value > best:
            best = value
            torch.save(dict(state=model.state_dict(), epoch=epoch, val_dice=best), folder / "best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"{folder.name}: epoch {epoch}/{args.epochs}, loss={history[-1]['train_loss']:.4f}, val Dice={value:.4f}", flush=True)
    write_csv(folder / "history.csv", history)
    checkpoint = torch.load(folder / "best.pt", map_location=images.device, weights_only=True)
    model.load_state_dict(checkpoint["state"])
    return model, checkpoint, time.perf_counter() - started


def summarize(output, records, originals, masks, args):
    runs, per_image = [], []
    for path in sorted((output / "checkpoints").glob("*/metrics.json")):
        runs.append(json.loads(path.read_text()))
    for path in sorted((output / "checkpoints").glob("*/test.csv")):
        with path.open(newline="") as stream:
            per_image.extend(csv.DictReader(stream))
    write_csv(output / "runs.csv", runs)
    write_csv(output / "per_image.csv", per_image)
    summary = []
    for cohort in ("original", "cleaned"):
        for method in args.methods:
            subset = [r for r in runs if r["cohort"] == cohort and r["method"] == method]
            row = dict(cohort=cohort, method=method, seeds=len(subset))
            for metric in ("clean_dice", "clean_iou", "lesion_dice", "normal_fp_rate", "native_dice", "native_iou"):
                values = [r[metric] for r in subset]
                row[metric] = float(np.mean(values))
                row[metric+"_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary.append(row)
    write_csv(output / "summary.csv", summary)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(args.methods))
    for offset, cohort in ((-0.18, "original"), (0.18, "cleaned")):
        rows = [r for r in summary if r["cohort"] == cohort]
        ax.bar(x+offset, [r["clean_dice"] for r in rows], 0.36, label=cohort,
               yerr=[r["clean_dice_seed_sd"] for r in rows], capsize=3)
    ax.set(xticks=x, xticklabels=args.methods, ylim=(0, 1), ylabel="Dice on common clean test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    # Choose examples from baseline scores once; display those same cases for every method.
    baseline = output / "checkpoints" / f"cleaned_raw_{args.seeds[0]}"
    if baseline.exists():
        with (baseline / "test.csv").open() as stream:
            entries = [r for r in csv.DictReader(stream) if r["clean"] == "True" and r["category"] != "normal"]
        entries.sort(key=lambda r: float(r["dice"]))
        chosen = entries[:2] + entries[-2:]
        fig, axes = plt.subplots(len(chosen), 2+len(args.methods), figsize=(16, 3*len(chosen)), squeeze=False)
        lookup = {r["image"]: i for i, r in enumerate(records)}
        for row_index, entry in enumerate(chosen):
            i = lookup[entry["image"]]
            axes[row_index, 0].imshow(originals[i], cmap="gray", vmin=0, vmax=255)
            axes[row_index, 0].set_title(entry["image"], fontsize=8)
            axes[row_index, 1].imshow(masks[i], cmap="gray", vmin=0, vmax=1)
            axes[row_index, 1].set_title("Ground truth")
            for col, method in enumerate(args.methods, 2):
                folder = output / "checkpoints" / f"cleaned_{method}_{args.seeds[0]}"
                with np.load(folder / "predictions.npz") as stored:
                    k = stored["indices"].tolist().index(i)
                    pred = stored["predictions"][k]
                axes[row_index, col].imshow(originals[i], cmap="gray", vmin=0, vmax=255)
                if pred.any():
                    axes[row_index, col].contour(pred, levels=[0.5], colors=["red"], linewidths=0.8)
                if masks[i].any():
                    axes[row_index, col].contour(masks[i], levels=[0.5], colors=["lime"], linewidths=0.8)
                axes[row_index, col].set_title(method)
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output / "examples.png", dpi=140)
        plt.close(fig)
    lines = ["# BUSI 前處理與資料清理比較", "", "本報告為實際執行結果，非文獻數值或模擬資料。", "",
             f"共 {len(runs)} 次訓練；每次 {args.epochs} epochs；訓練種子 {args.seeds}。",
             "兩版本均在相同的乾淨驗證集選最佳 epoch，以下主表使用相同乾淨測試集。", "",
             "| 訓練資料 | 前處理 | Dice（種子平均 ± SD） | IoU（種子平均 ± SD） | 病灶 Dice | 正常影像偽陽性率 |",
             "|---|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['cohort']} | {r['method']} | {r['clean_dice']:.4f} ± {r['clean_dice_seed_sd']:.4f} | {r['clean_iou']:.4f} ± {r['clean_iou_seed_sd']:.4f} | {r['lesion_dice']:.4f} | {r['normal_fp_rate']:.4f} |")
    lines.extend(["", "![共用乾淨測試集比較](comparison.png)", "", "## 清理的配對差異", "",
                  "同一方法、同一種子配對：cleaned − original；單位為 Dice 百分點。", ""])
    for method in args.methods:
        deltas = []
        for seed in args.seeds:
            paired = {r["cohort"]: r for r in runs if r["method"] == method and r["seed"] == seed}
            deltas.append(100*(paired["cleaned"]["clean_dice"] - paired["original"]["clean_dice"]))
        lines.append(f"- {method}: 平均 {np.mean(deltas):+.2f}；各種子 {', '.join(f'{v:+.2f}' for v in deltas)}")
    lines.extend(["", "## 成功與失敗案例", "", "固定取 cleaned/raw 第一個種子的病灶測試集兩個最低與兩個最高 Dice 案例；紅線為預測，綠線為標註。其他方法展示相同影像，未各自挑選最佳案例。", "", "![預測案例](examples.png)", "",
                  "## 解讀限制", "", "- 使用作者勘誤群組及完全相同像素雜湊分組；沒有病人識別碼，不能宣稱病人層級獨立。",
                  "- 清理版排除作者標記 axilla、needle、multiclass 群組，重複群組保留清單第一張；保留正常影像與含文字／量測標記的影像。未重新請醫師標註。",
                  "- 原版保留全部 780 張，但採分組切分；不是會讓重複影像跨區的天真隨機基準。",
                  "- 兩版 epoch 數一致，但訓練資料量不同，因此最佳化更新次數不同；效果同時包含資料清理與樣本數變化。",
                  "- SD 是固定一次切分下的訓練種子變異，不是病人層級信賴區間或顯著性證據。",
                  "- 閾值固定 0.5；空標註且空預測的 Dice/IoU 定義為 1；正常影像只要預測任一陽性像素即計為偽陽性。",
                  "- 未做超參數搜尋、交叉驗證或外部資料驗證；不作臨床診斷用途。",
                  "", "詳細設定見 config.json、資料數量見 dataset_stats.json、逐張結果見 per_image.csv。",
                  "", "資料清理依據：[Pawłowska 等人補充資料（CC BY 4.0）](https://doi.org/10.17632/k8t3gnx9h6.1)。"])
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/Dataset_BUSI_with_GT"))
    parser.add_argument("--output", type=Path, default=Path("results/busi"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    if args.epochs < 1 or args.size < 32 or args.size % 16 or args.batch_size < 1:
        parser.error("epochs/batch-size must be positive; size must be >=32 and divisible by 16")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate seeds/methods are not allowed")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_records(args.data, Path("metadata/dataset_comment_list.csv"))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = dict(epochs=args.epochs, seeds=args.seeds, size=args.size, batch_size=args.batch_size,
                  methods=args.methods, split_seed=2026, learning_rate=0.001, threshold=0.5,
                  median_kernel=3, clahe_clip=2.0, clahe_grid=[8, 8], srad_iterations=20,
                  srad_step=0.5, srad_order="CLAHE then SRAD", augmentation="rotation +/-10 deg, scale 0.95-1.05, zero padding",
                  model="U-Net base16 four downsamplings", loss="BCE + soft Dice", validation="common cleaned val",
                  device=str(device), gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__,
                  manifest_sha256=hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in ("experiment.py", "dataset.py", "preprocessing.py", "model.py")})
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output contains a different experiment; select a new --output directory")
    config_path.write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    write_csv(output / "manifest.csv", [{**r, "masks": "|".join(r["masks"])} for r in records])
    stats = {cohort: {split: dict(Counter(r["category"] for r in records if r["split"] == split and (cohort == "original" or r["clean"])))
                      for split in ("train", "val", "test")} for cohort in ("original", "cleaned")}
    stats["exclusion_reasons"] = dict(Counter(r["exclusion"] for r in records if not r["clean"]))
    stats["groups"] = len({r["group"] for r in records})
    (output / "dataset_stats.json").write_text(json.dumps(stats, indent=2)+"\n")
    print(json.dumps(stats), flush=True)
    originals, target = load_arrays(args.data, records, args.size)
    masks = torch.from_numpy(target[:, None].astype(np.float32)).to(device, memory_format=torch.channels_last)
    val_indices = torch.tensor([i for i, r in enumerate(records) if r["split"] == "val" and r["clean"]], device=device)
    test_list = [i for i, r in enumerate(records) if r["split"] == "test"]
    test_indices = torch.tensor(test_list, device=device)
    clean_selector = np.array([records[i]["clean"] for i in test_list])
    lesion_selector = np.array([records[i]["category"] != "normal" for i in test_list]) & clean_selector
    normal_selector = ~np.array([records[i]["category"] != "normal" for i in test_list]) & clean_selector
    for method in args.methods:
        print(f"Preprocessing {method} on {device}", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            processed = np.stack(list(pool.map(lambda image: preprocess(image, method), originals)))
        images = torch.from_numpy(processed[:, None].astype(np.float32)/255).to(device, memory_format=torch.channels_last)
        for cohort in ("original", "cleaned"):
            indices = torch.tensor([i for i, r in enumerate(records) if r["split"] == "train" and (cohort == "original" or r["clean"])], device=device)
            for seed in args.seeds:
                folder = output / "checkpoints" / f"{cohort}_{method}_{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                if (folder / "metrics.json").exists():
                    print(f"Already complete: {folder.name}", flush=True)
                    continue
                model, best, seconds = train(images, masks, indices, val_indices, args, seed, folder)
                pred, dice, iou = predict(model, images, masks, test_indices, args.batch_size)
                native = np.ones(len(test_list), dtype=bool) if cohort == "original" else clean_selector
                metrics = dict(cohort=cohort, method=method, seed=seed, best_epoch=best["epoch"],
                               val_dice=best["val_dice"], train_n=len(indices), clean_test_n=int(clean_selector.sum()),
                               native_test_n=int(native.sum()), train_seconds=seconds,
                               clean_dice=float(dice[clean_selector].mean()), clean_iou=float(iou[clean_selector].mean()),
                               clean_dice_image_sd=float(dice[clean_selector].std(ddof=1)),
                               lesion_dice=float(dice[lesion_selector].mean()),
                               normal_fp_rate=float(pred[normal_selector].any((1, 2)).mean()),
                               native_dice=float(dice[native].mean()), native_iou=float(iou[native].mean()))
                write_csv(folder / "test.csv", [dict(cohort=cohort, method=method, seed=seed,
                          image=records[i]["image"], category=records[i]["category"], clean=records[i]["clean"],
                          dice=float(dice[k]), iou=float(iou[k])) for k, i in enumerate(test_list)])
                if seed == args.seeds[0]:
                    np.savez_compressed(folder / "predictions.npz", indices=test_list, predictions=pred)
                (folder / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
                print(f"Completed {folder.name}: clean Dice={metrics['clean_dice']:.4f}, {seconds:.1f}s", flush=True)
                del model
        del images
    summarize(output, records, originals, target, args)
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
