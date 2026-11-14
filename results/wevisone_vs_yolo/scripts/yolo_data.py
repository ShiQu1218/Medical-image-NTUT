"""Export sources.py specs to Ultralytics YOLO-seg format (one polygon line per contour).

Normal images get an empty label file, so YOLO trains on them as pure background.
  python yolo_data.py --out ../data_yolo/busi --train busi:train --val busi:val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import common
import sources


def polygons(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < 4 or len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, 0.0005 * cv2.arcLength(c, True), True)
        if len(c) >= 3:
            out.append(c[:, 0, :])
    return out


def export(splits: dict[str, str], out: Path) -> Path:
    """splits: yolo split name -> sources spec. Returns data.yaml path."""
    yaml = out / "data.yaml"
    if yaml.exists() and json.loads((out / "manifest.json").read_text()).get("specs") == splits:
        return yaml
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(specs=splits)
    for split, spec in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for it in sources.items(spec):
            stem = it["id"].replace("/", "__").rsplit(".", 1)[0]
            img_path = out / "images" / split / f"{stem}.png"
            cv2.imencode(".png", np.asarray(it["image"].convert("L")))[1].tofile(str(img_path))
            h, w = it["union"].shape
            lines = []
            for m in it["instances"]:
                for poly in polygons(m):
                    xy = (poly / np.array([w, h])).clip(0, 1).flatten()
                    lines.append("0 " + " ".join(f"{v:.6f}" for v in xy))
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
            n += 1
        manifest[split] = n
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    yaml.write_text(f"path: {out.resolve().as_posix()}\n" + "".join(f"{s}: images/{s}\n" for s in splits)
                    + "names:\n  0: lesion\n")
    return yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    args = ap.parse_args()
    yaml = export({"train": args.train, "val": args.val}, Path(args.out))
    print(yaml, (yaml.parent / "manifest.json").read_text())


if __name__ == "__main__":
    main()
