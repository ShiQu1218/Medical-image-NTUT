"""Build nnU-Net v2 raw datasets from sources.py specs (2D PNG, labels 0/1).

Dataset501_BUSIclean : imagesTr = busi:train + busi:val (incl. normals with empty masks);
                       splits_final.json fold 0 = exactly our train/val split.
Dataset502_BUSBRA    : imagesTr = busbra:train + busbra:val; fold 0 = that split (for BUS-BRA -> BUSI transfer).
Test images go to imagesTs_<spec key>/ with a cases.json mapping case id -> sources id.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

import common
import sources

NN = common.ROOT / "nnunet"
os.environ.setdefault("nnUNet_raw", str(NN / "raw"))
os.environ.setdefault("nnUNet_preprocessed", str(NN / "preprocessed"))
os.environ.setdefault("nnUNet_results", str(NN / "results"))

DATASETS = {
    501: dict(name="BUSIclean", train="busi:train", val="busi:val",
              tests=["busi:val", "busi:test", "uclm:all", "busuc:all", "busbra:test", "breast:all"]),
    502: dict(name="BUSBRA", train="busbra:train", val="busbra:val",
              tests=["busbra:val", "busbra:test", "busirand0:train+busirand0:val+busirand0:test", "uclm:all"]),
}


def case_id(prefix, i):
    return f"{prefix}_{i:05d}"


def write_png(path: Path, arr: np.ndarray):
    cv2.imencode(".png", arr)[1].tofile(str(path))


def build(did: int, tests_only=False):
    cfg = DATASETS[did]
    root = NN / "raw" / f"Dataset{did}_{cfg['name']}"
    if tests_only:
        return build_tests(root, cfg["tests"])
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    cases, split = {}, {"train": [], "val": []}
    i = 0
    for part in ("train", "val"):
        for it in sources.items(cfg[part]):
            cid = case_id("tr", i)
            i += 1
            write_png(root / "imagesTr" / f"{cid}_0000.png", np.asarray(it["image"].convert("L")))
            write_png(root / "labelsTr" / f"{cid}.png", (it["union"] > 0).astype(np.uint8))
            cases[cid] = it["id"]
            split[part].append(cid)
    (root / "dataset.json").write_text(json.dumps(dict(
        channel_names={"0": "US"}, labels={"background": 0, "lesion": 1}, numTraining=i, file_ending=".png"), indent=2))
    (root / "cases_tr.json").write_text(json.dumps(cases, indent=2))
    pre = NN / "preprocessed" / f"Dataset{did}_{cfg['name']}"
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "splits_final.json").write_text(json.dumps([split], indent=2))
    build_tests(root, cfg["tests"])
    print(did, "train", len(split["train"]), "val", len(split["val"]))


def build_tests(root, tests):
    for spec in tests:
        d = root / f"imagesTs_{spec.replace(':', '_').replace('+', '-')}"
        if (d / "cases.json").exists():
            continue
        d.mkdir(exist_ok=True)
        tcases = {}
        for j, it in enumerate(sources.items(spec)):
            cid = case_id("ts", j)
            write_png(d / f"{cid}_0000.png", np.asarray(it["image"].convert("L")))
            tcases[cid] = it["id"]
        (d / "cases.json").write_text(json.dumps(dict(spec=spec, cases=tcases), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="501,502")
    ap.add_argument("--tests_only", action="store_true")
    ap.add_argument("--extra_tests", default="", help="comma list of extra test specs (e.g. corruptions)")
    a = ap.parse_args()
    for did in map(int, a.ids.split(",")):
        if a.extra_tests:
            build_tests(NN / "raw" / f"Dataset{did}_{DATASETS[did]['name']}", a.extra_tests.split(","))
        else:
            build(did, a.tests_only)
