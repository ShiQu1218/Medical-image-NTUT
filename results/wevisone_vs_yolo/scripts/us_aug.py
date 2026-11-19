"""Ultrasound photometric augmentation for YOLO training, matching train.py strong_aug (photometric part).

Geometric part (scale 0.8-1.2, rotate ±10°, translate ±10%) is covered by Ultralytics' own random_perspective
(scale=0.5, translate=0.1, degrees=10 when --aug strong). Applied after mosaic/affine, before Format.
"""
import random

import cv2
import numpy as np


class UltrasoundAug:
    def __call__(self, labels):
        img = labels["img"]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 if img.ndim == 3 else img.astype(np.float32) / 255.0
        h, w = g.shape
        if random.random() < 0.8:  # gamma, contrast, brightness
            g = np.clip(g, 0, 1) ** random.uniform(0.6, 1.6)
            g = (g - g.mean()) * random.uniform(0.75, 1.3) + g.mean() + random.uniform(-0.1, 0.1)
        if random.random() < 0.3:  # multiplicative speckle
            L = random.uniform(8, 30)
            g = g * np.random.gamma(L, 1.0 / L, g.shape).astype(np.float32)
        if random.random() < 0.25:
            k = random.choice([3, 5])
            g = cv2.GaussianBlur(g, (k, k), 0)
        if random.random() < 0.2:  # resolution loss
            f = random.uniform(0.4, 0.8)
            g = cv2.resize(cv2.resize(g, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA), (w, h))
        g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
        labels["img"] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if img.ndim == 3 else g
        return labels


def install():
    from ultralytics.data.dataset import YOLODataset
    if getattr(YOLODataset, "_us_aug", False):
        return
    orig = YOLODataset.build_transforms

    def build_transforms(self, hyp):
        t = orig(self, hyp)
        if self.augment:
            t.insert(len(t.transforms) - 1, UltrasoundAug())
        return t
    YOLODataset.build_transforms = build_transforms
    YOLODataset._us_aug = True
