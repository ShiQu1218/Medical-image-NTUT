"""Fixed-update validation experiments. Never evaluates the test partition."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import load_records, load_arrays
from experiment import augment, loss_fn, predict, write_csv
from model import UNet
from preprocessing import METHODS, preprocess


def matched_training_indices(records, seed):
    """Sample only original training cases, matching cleaned class counts."""
    rng = np.random.default_rng(seed)
    selected = []
    for category in ("benign", "malignant", "normal"):
        pool = [i for i,r in enumerate(records) if r["split"] == "train" and r["category"] == category]
        count = sum(records[i]["clean"] for i in pool)
        selected.extend(rng.choice(pool, size=count, replace=False).tolist())
    return sorted(selected)


def run(images, masks, indices, val, config, seed, prefix):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = UNet().to(images.device, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    history, best, step, epoch = [], -1., 0, 0
    start = time.perf_counter()
    while step < config["budgets"][-1]:
        epoch += 1
        order = indices[torch.randperm(len(indices), device=images.device)]
        for batch in order.split(config["batch_size"]):
            model.train()
            x, y = augment(images[batch], masks[batch])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            step += 1
            if step % config["validate_every"] == 0 or step in config["budgets"]:
                pred, dice, iou = predict(model, images, masks, val, config["batch_size"])
                score = float(dice.mean())
                history.append(dict(step=step, epoch=epoch, batch_loss=loss.item(), val_dice=score))
                if score > best:
                    best, best_step = score, step
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    best_iou = float(iou.mean())
                if step % 350 == 0:
                    print(f"{prefix.name}: step={step}, val={score:.4f}, best={best:.4f}", flush=True)
            if step in config["budgets"]:
                folder = prefix.parent / f"{prefix.name}_steps{step}"
                folder.mkdir(parents=True, exist_ok=True)
                torch.save(dict(state=best_state, step=best_step, val_dice=best), folder / "best.pt")
                write_csv(folder / "history.csv", history)
                metrics = dict(cohort=prefix.name.split("_")[0], method=prefix.name.split("_", 1)[1].rsplit("_", 1)[0],
                               seed=seed, updates=step, train_n=len(indices), val_n=len(val),
                               best_step=best_step, val_dice=best, val_iou=best_iou,
                               elapsed_seconds=time.perf_counter()-start)
                (folder / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
            if step == config["budgets"][-1]:
                break


def summarize(output, config):
    rows = [json.loads(p.read_text()) for p in sorted((output / "checkpoints").glob("*/metrics.json"))]
    assert len(rows) == 2*len(config["methods"])*len(config["seeds"])*len(config["budgets"])
    write_csv(output / "runs.csv", rows)
    summary = []
    for budget in config["budgets"]:
        for cohort in config.get("cohorts", ("original", "cleaned")):
            for method in config["methods"]:
                group = [r for r in rows if (r["updates"], r["cohort"], r["method"]) == (budget, cohort, method)]
                values = [r["val_dice"] for r in group]
                summary.append(dict(updates=budget, cohort=cohort, method=method,
                                    val_dice=float(np.mean(values)), seed_sd=float(np.std(values, ddof=1))))
    write_csv(output / "summary.csv", summary)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, budget in zip(axes, config["budgets"]):
        x = np.arange(len(config["methods"]))
        for cohort, offset in zip(config.get("cohorts", ("original", "cleaned")), (-.18, .18)):
            subset = [r for r in summary if r["updates"] == budget and r["cohort"] == cohort]
            ax.bar(x+offset, [r["val_dice"] for r in subset], .36,
                   yerr=[r["seed_sd"] for r in subset], label=cohort, capsize=3)
        ax.set(xticks=x, xticklabels=config["methods"], title=f"{budget} updates", ylim=(0, 1), ylabel="Best validation Dice")
        ax.tick_params(axis="x", labelrotation=15)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    lines = ["# 等更新次數與延長訓練：驗證集結果", "", "本輪未執行測試集推論；以下都是用來選模型的驗證集分數，不能視為測試效能。", "",
             "| 更新次數 | 訓練版本 | 方法 | 最佳驗證 Dice（種子平均 ± SD） |", "|---:|---|---|---:|"]
    for row in summary:
        lines.append(f"| {row['updates']} | {row['cohort']} | {row['method']} | {row['val_dice']:.4f} ± {row['seed_sd']:.4f} |")
    lines += ["", "![比較圖](comparison.png)", "", "參數見 [parameters.md](parameters.md)、config.json；逐次訓練見 runs.csv 與 checkpoints/ 下的 history.csv。",
              "", "固定更新次數不代表完全相同的影像曝光量：保留各 epoch 最後一個不足 batch 的批次。2,800 次仍是預算上限，不保證已收斂。增加驗證次數也會增加選模型的機會，1,400／2,800 分數不是獨立重複實驗。"]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/followup"))
    parser.add_argument("--matched", action="store_true", help="Match original training sample and class counts to cleaned")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = dict(budgets=[1400, 2800], validate_every=35, batch_size=16, lr=.001,
                  cohorts=["matched", "cleaned"] if args.matched else ["original", "cleaned"], sample_seed=2026,
                  seeds=[42,43,44], methods=list(METHODS), size=256, split_seed=2026,
                  test_evaluated=False, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  sources={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                           for name in ("followup.py", "experiment.py", "dataset.py", "model.py", "preprocessing.py")})
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Different experiment exists; use a new output directory")
    path.write_text(json.dumps(config, indent=2)+"\n")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    records = load_records(Path("data/Dataset_BUSI_with_GT"), Path("metadata/dataset_comment_list.csv"))
    # Test images are excluded before loading arrays or preprocessing.
    records = [r for r in records if r["split"] == "train" or (r["split"] == "val" and r["clean"])]
    sampled = matched_training_indices(records, config["sample_seed"]) if args.matched else []
    if args.matched:
        write_csv(output / "sampled_original.csv", [{**records[i], "masks": "|".join(records[i]["masks"])} for i in sampled])
    write_csv(output / "manifest.csv", [{**r, "masks": "|".join(r["masks"])} for r in records])
    raw, target = load_arrays(Path("data/Dataset_BUSI_with_GT"), records, config["size"])
    masks = torch.from_numpy(target[:,None].astype(np.float32)).to("cuda", memory_format=torch.channels_last)
    val = torch.tensor([i for i,r in enumerate(records) if r["split"] == "val"], device="cuda")
    for method in METHODS:
        with ThreadPoolExecutor(max_workers=4) as pool:
            processed = np.stack(list(pool.map(lambda image: preprocess(image, method), raw)))
        images = torch.from_numpy(processed[:,None].astype(np.float32)/255).to("cuda", memory_format=torch.channels_last)
        for cohort in config["cohorts"]:
            selected = sampled if cohort == "matched" else [i for i,r in enumerate(records) if r["split"] == "train" and (cohort == "original" or r["clean"])]
            indices = torch.tensor(selected, device="cuda")
            for seed in config["seeds"]:
                prefix = output / "checkpoints" / f"{cohort}_{method}_{seed}"
                if all((prefix.parent / f"{prefix.name}_steps{b}" / "metrics.json").exists() for b in config["budgets"]):
                    continue
                run(images, masks, indices, val, config, seed, prefix)
        del images
    summarize(output, config)
    print("Completed follow-up", flush=True)


if __name__ == "__main__":
    main()
