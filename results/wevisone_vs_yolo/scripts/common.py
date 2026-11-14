"""Shared BUSI data, prompt/target formatting, metrics, and model loading."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NTUT = ROOT / "Medical-image-NTUT"
WEVIS = ROOT / "WeVisOne-VL"
DATA = NTUT / "data" / "Dataset_BUSI_with_GT"
MODEL_DIR = Path.home() / "models" / "WeVisOne-VL-4B"
RUNS = ROOT / "runs"

sys.path.insert(0, str(WEVIS))
sys.path.insert(0, str(NTUT))
from dataset import load_records, read_gray  # noqa: E402  (user's repo: same split as U-Net baseline)
from wevisone_vl.prompts import build_prompt  # noqa: E402

QUERY = "breast lesion"
PROMPT = build_prompt("segmentation", QUERY)
EVAL_SIZE = 256  # U-Net baseline resolution


def clean_split(split: str) -> list[dict]:
    """Cleaned cohort rows for train/val/test (368/85/76), identical to the repo split."""
    records = load_records(DATA, NTUT / "metadata" / "dataset_comment_list.csv")
    return [r for r in records if r["split"] == split and r["clean"]]


def load_sample(row: dict):
    """RGB PIL image, list of per-instance binary masks (H,W uint8), union mask."""
    gray = read_gray(DATA / row["image"])
    image = Image.fromarray(gray).convert("RGB")
    instances = []
    for rel in row["masks"]:
        m = (read_gray(DATA / rel) > 0).astype(np.uint8)
        if m.any():
            instances.append(m)
    union = np.zeros_like(gray, dtype=np.uint8)
    for m in instances:
        union |= m
    return image, instances, union


def tight_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    return [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]


def target_response(boxes: list[list[float]]) -> str:
    if not boxes:
        return "None"
    # Native multi-instance format observed zero-shot: one <box> with comma-separated entries.
    parts = ["".join(f"<{max(0, min(1000, round(v * 1000)))}>" for v in box) + "<OBJ>" for box in boxes]
    return f"<ref>{QUERY}</ref><box>{','.join(parts)}</box>"


def scores(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Same definition as Medical-image-NTUT experiment.scores (empty/empty = 1)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    union = (pred | gt).sum()
    dice = 1.0 if total == 0 else 2 * inter / total
    iou = 1.0 if union == 0 else inter / union
    return float(dice), float(iou)


def to_eval_res(prob: np.ndarray | None, union: np.ndarray, threshold: float = 0.5):
    """Resize prediction/GT to 256x256 like the baseline: GT nearest, prob bilinear."""
    gt = cv2.resize(union, (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_NEAREST)
    if prob is None:
        return np.zeros_like(gt), gt
    p = cv2.resize(prob.astype(np.float32), (EVAL_SIZE, EVAL_SIZE), interpolation=cv2.INTER_LINEAR)
    return (p > threshold).astype(np.uint8), gt


def load_engine():
    from wevisone_vl import WeVisOneVL

    return WeVisOneVL(str(MODEL_DIR), device="cuda:0", dtype="bfloat16")
