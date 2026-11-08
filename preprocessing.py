"""Grayscale preprocessing; masks must never enter these functions."""
import cv2
import numpy as np

METHODS = ("raw", "clahe", "srad", "median", "clahe_srad")


def srad(image, iterations=20, step=0.5):
    """Yu & Acton (2002), bounded conduction, symmetric no-flux stencil.

    shortcut: q0 is estimated from the whole image, use a validated homogeneous
    ROI estimator if homogeneous tissue regions become available.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("SRAD expects a 2-D uint8 image")
    if iterations < 0 or not 0 < step <= 1:
        raise ValueError("iterations must be nonnegative and 0 < step <= 1")
    offset = 1 / 255
    u = image.astype(np.float32) / 255 + offset
    for _ in range(iterations):
        p = np.pad(u, 1, mode="edge")
        n, s = p[:-2, 1:-1] - u, p[2:, 1:-1] - u
        w, e = p[1:-1, :-2] - u, p[1:-1, 2:] - u
        gradient = (n*n + s*s + w*w + e*e) / (u*u)
        laplacian = (n + s + w + e) / u
        q2 = np.maximum(0, (0.5*gradient - 0.0625*laplacian**2)
                        / ((1 + 0.25*laplacian)**2 + 1e-12))
        q02 = float(u.var() / (u.mean()**2 + 1e-12))
        if q02 < 1e-12:
            break
        c = np.clip(1 / (1 + (q2 - q02) / (q02*(1 + q02)) + 1e-12), 0, 1)
        cp = np.pad(c, 1, mode="edge")
        flux = ((c + cp[:-2, 1:-1])*n + (c + cp[2:, 1:-1])*s
                + (c + cp[1:-1, :-2])*w + (c + cp[1:-1, 2:])*e) / 2
        u += step / 4 * flux
    return np.clip(np.rint((u - offset)*255), 0, 255).astype(np.uint8)


def preprocess(image, method):
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected a 2-D uint8 image")
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if method == "median":
        return cv2.medianBlur(image, 3)
    if method in ("clahe", "clahe_srad"):
        image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    if method in ("srad", "clahe_srad"):
        image = srad(image)
    return image.copy()
