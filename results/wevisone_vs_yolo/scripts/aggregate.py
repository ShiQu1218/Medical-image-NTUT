"""Mean ± sample SD across seeds of test summaries for configs A and B."""
import json

import numpy as np

import common

R = common.RUNS
configs = {
    "A (original)": [R / "test_full", R / "sweep/A_seed43/test", R / "sweep/A_seed44/test"],
    "B (normals x2)": [R / f"sweep/B_seed{s}/test" for s in (42, 43, 44)],
}
keys = ["dice", "iou", "lesion_dice", "normal_fp_rate", "lesion_miss_rate"]
print("| config | n | " + " | ".join(keys) + " | gt_box lesion_dice |")
print("|---|---:|" + "---:|" * (len(keys) + 1))
for name, dirs in configs.items():
    gen = [json.loads((d / "test_generate_summary.json").read_text()) for d in dirs if (d / "test_generate_summary.json").exists()]
    box = [json.loads((d / "test_gt_box_summary.json").read_text()) for d in dirs if (d / "test_gt_box_summary.json").exists()]
    cells = []
    for k in keys:
        v = np.array([g[k] for g in gen])
        cells.append(f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}")
    b = np.array([g["lesion_dice"] for g in box])
    cells.append(f"{b.mean():.4f} ± {b.std(ddof=1):.4f}" if len(b) > 1 else f"{b.mean():.4f}")
    print(f"| {name} | {len(gen)} | " + " | ".join(cells) + " |")
    print("  per-seed normal_fp:", [round(g["normal_fp_rate"], 2) for g in gen],
          " per-seed dice:", [round(g["dice"], 4) for g in gen])
