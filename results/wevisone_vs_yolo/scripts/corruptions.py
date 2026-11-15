"""Synthetic ultrasound domain-shift suite (stress test, ImageNet-C style). Parameters are fixed a priori.

Each corruption is deterministic per (image id, family, severity). Geometric ones transform the GT too.
Families marked IN_TRAIN overlap the 'strong' training augmentation of train.py (report them separately).

  gain     global gain/gamma change                       (IN_TRAIN)
  speckle  multiplicative gamma speckle                   (IN_TRAIN)
  blur     resolution loss + blur                         (IN_TRAIN)
  tgc      depth-dependent gain ramp (time-gain comp.)
  jpeg     JPEG re-compression
  icon     body-mark pictogram + text overlays in corners (the UCLM shortcut)
  sector   convex-probe fan: everything outside a sector is black (GT cut accordingly)
  fov      zoom-out: image shrunk inside black UI frame (field of view / scanner layout)

Memorization probe (memo_probe.py; not part of FAMILIES): mild label-preserving geometry, GT transformed too.
  hflip    horizontal mirror (sev 1 only; also in the fine-tuning flip augmentation)
  rot      rotation +-8 / +-15 deg
  zoom     centre zoom-in x1.15 / x1.3
  shift    translation 8% / 15% of width and height (black border)
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

FAMILIES = ["gain", "speckle", "blur", "tgc", "jpeg", "icon", "sector", "fov"]
IN_TRAIN = {"gain", "speckle", "blur"}


def _rng(key: str):
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))


def _icon(g, rng, sev):
    h, w = g.shape
    out = g.copy()
    s = int(min(h, w) * (0.10 if sev == 1 else 0.16))
    for k in range(1 if sev == 1 else 2):  # pictogram: breast outline + probe bar, like scanner body marks
        x0 = w - s - 8 if k == 0 else 8
        y0 = h - s - 8
        cv2.rectangle(out, (x0, y0), (x0 + s, y0 + s), 90, -1)
        cv2.ellipse(out, (x0 + s // 2, y0 + s // 2), (s // 3, s // 3), 0, 0, 180, 230, 2)
        cv2.circle(out, (x0 + s // 2, y0 + s // 2), max(2, s // 14), 230, -1)
        a = rng.uniform(0, np.pi)
        cx, cy = x0 + s // 2, y0 + s // 2
        cv2.line(out, (int(cx - s * 0.35 * np.cos(a)), int(cy - s * 0.35 * np.sin(a))),
                 (int(cx + s * 0.35 * np.cos(a)), int(cy + s * 0.35 * np.sin(a))), 255, 3)
    txt = ["R BREAST 10:00 3CM FN", "L BR RAD", "LT 2:00 4CMFN"][int(rng.integers(3))]
    cv2.putText(out, txt, (8, int(h * 0.06) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5 if sev == 1 else 0.8, 240, 1 if sev == 1 else 2)
    if sev == 2:
        cv2.putText(out, "5.0cm  G62  DR70", (w - 200, int(h * 0.06) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 240, 1)
    return out


def apply(gray: np.ndarray, masks: list[np.ndarray], family: str, sev: int, key: str):
    """gray uint8 HxW, masks list of uint8 HxW -> (gray', masks')."""
    rng = _rng(f"{key}|{family}|{sev}")
    g = gray.astype(np.float32) / 255.0
    h, w = g.shape
    if family == "gain":
        g = np.clip(g, 0, 1) ** (1.7 if sev == 1 else 2.3) if rng.random() < 0.5 else np.clip(g, 0, 1) ** (0.6 if sev == 1 else 0.45)
    elif family == "speckle":
        L = 6 if sev == 1 else 2.5
        g = g * rng.gamma(L, 1.0 / L, g.shape).astype(np.float32)
    elif family == "blur":
        f = 0.5 if sev == 1 else 0.3
        g = cv2.resize(cv2.resize(g, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA), (w, h))
        g = cv2.GaussianBlur(g, (0, 0), 1.5 if sev == 1 else 3.0)
    elif family == "tgc":
        depth = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        up = rng.random() < 0.5
        a = 0.6 if sev == 1 else 0.85
        ramp = 1 - a * depth if up else (1 - a) + a * depth
        g = g * ramp
    elif family == "jpeg":
        q = 25 if sev == 1 else 8
        ok, buf = cv2.imencode(".jpg", (np.clip(g, 0, 1) * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
        g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    elif family == "icon":
        return _icon(gray, rng, sev), masks
    elif family == "sector":
        apex = (w / 2, -h * (0.25 if sev == 1 else 0.10))
        half = np.deg2rad(50 if sev == 1 else 40)
        yy, xx = np.mgrid[0:h, 0:w]
        ang = np.arctan2(xx - apex[0], yy - apex[1])
        fan = (np.abs(ang) <= half).astype(np.uint8)
        g = g * fan
        masks = [m * fan for m in masks]
    elif family == "fov":
        s = 0.75 if sev == 1 else 0.55
        nw, nh = int(w * s), int(h * s)
        x0, y0 = (w - nw) // 2 + int(rng.integers(-w // 20, w // 20 + 1)), int(h * 0.08)
        canvas = np.zeros_like(g)
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
        g = canvas
        new = []
        for m in masks:
            c = np.zeros_like(m)
            c[y0:y0 + nh, x0:x0 + nw] = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
            new.append(c)
        masks = new
    elif family in ("hflip", "rot", "zoom", "shift"):  # memorization probe: mild, label-preserving geometry
        if family == "hflip":
            M = np.float32([[-1, 0, w - 1], [0, 1, 0]])
        elif family == "rot":
            M = cv2.getRotationMatrix2D((w / 2, h / 2), (8 if sev == 1 else 15) * (1 if rng.random() < 0.5 else -1), 1.0)
        elif family == "zoom":  # zoom-in around the centre (edges cropped)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.15 if sev == 1 else 1.3)
        else:
            f = 0.08 if sev == 1 else 0.15
            M = np.float32([[1, 0, f * w * (1 if rng.random() < 0.5 else -1)], [0, 1, f * h * (1 if rng.random() < 0.5 else -1)]])
        g = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        masks = [cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0) for m in masks]
    else:
        raise ValueError(family)
    return (np.clip(g, 0, 1) * 255).astype(np.uint8), [m for m in masks if m.sum() >= 16]
