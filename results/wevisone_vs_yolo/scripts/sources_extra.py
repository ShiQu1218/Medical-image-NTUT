"""Extra datasets registered into sources.EXTRA.

busbra:train|val|test|all  BUS-BRA (Zenodo 8231412), 1875 lesion images / 1064 patients, split 70/10/20 by Case
busuc:all                  BUS_UC (Mendeley 3ksd7w7jkx), 811 lesion images at 256x256, no patient ids (1 image = 1 group)
Neither has normal images.
"""
from __future__ import annotations

import csv

import cv2
import numpy as np
from PIL import Image

import common
import sources

BRA = common.ROOT / "external" / "BUS-BRA" / "data" / "BUSBRA"
UC = common.ROOT / "external" / "BUS_UC"
UC_FOLDERS = {"benign": ("151e9be2-b909-4431-add0-e64a1345995f", "abaf19cb-82c4-4988-bed0-515299d663e7"),
              "malignant": ("162b134c-e9d9-4996-9c23-cfaa57d110d8", "1a9e79b7-c492-40b6-a023-7ea8c5dbc716")}


def _read(path, flags=cv2.IMREAD_GRAYSCALE):
    a = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    if a is None:
        raise ValueError(path)
    return a


def _busbra(split):
    with (BRA / "bus_data.csv").open(encoding="utf-8-sig") as f:
        rows = [dict(r, _src="busbra", category=r["Pathology"].lower(), patient=f"bra{r['Case']}") for r in csv.DictReader(f)]
    if split == "all":
        return rows
    bins = sources._stratified_groups(rows, "patient", 10, seed=2026)
    want = {"train": range(0, 7), "val": range(7, 8), "test": range(8, 10)}[split]
    return [r for r in rows if bins[r["patient"]] in want]


def _busbra_load(row):
    gray = _read(BRA / "Images" / f"{row['ID']}.png")
    mask = (_read(BRA / "Masks" / f"mask_{row['ID'][4:]}.png") > 127).astype(np.uint8)
    if mask.shape != gray.shape:
        mask = cv2.resize(mask, gray.shape[::-1], interpolation=cv2.INTER_NEAREST)
    return dict(id=f"busbra/{row['ID']}", image=Image.fromarray(gray).convert("RGB"), union=mask,
                instances=sources.components(mask), category=row["category"], patient=row["patient"],
                source="busbra", device=row["Device"])


def _busuc(split):
    if split != "all":
        raise ValueError("BUS_UC is external-test only")
    rows = []
    for cat, (img_dir, mask_dir) in UC_FOLDERS.items():
        for p in sorted((UC / img_dir).iterdir()):
            m = UC / mask_dir / p.name
            if not m.exists():
                raise ValueError(f"missing mask {m}")
            rows.append(dict(_src="busuc", category=cat, name=p.name, img=str(p), mask=str(m),
                             patient=f"uc_{cat}_{p.stem}"))
    return rows


def _busuc_load(row):
    gray = _read(row["img"])
    mask = (_read(row["mask"]) > 127).astype(np.uint8)
    return dict(id=f"busuc/{row['category']}/{row['name']}", image=Image.fromarray(gray).convert("RGB"), union=mask,
                instances=sources.components(mask), category=row["category"], patient=row["patient"], source="busuc")


def _busirand(split, seed=0):
    """Replicates Yu et al. 2026 (QIMS): ALL original BUSI benign+malignant images (no cleaning, no
    duplicate grouping), image-level random 80/10/10 split stratified by class."""
    recs = [dict(r, _src="busi") for r in common.load_records(common.DATA, common.NTUT / "metadata" / "dataset_comment_list.csv")
            if r["category"] != "normal"]
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for cat in ("benign", "malignant"):
        rows = sorted((r for r in recs if r["category"] == cat), key=lambda r: r["image"])
        idx = rng.permutation(len(rows))
        n_tr, n_va = int(round(0.8 * len(rows))), int(round(0.1 * len(rows)))
        out["train"] += [rows[i] for i in idx[:n_tr]]
        out["val"] += [rows[i] for i in idx[n_tr:n_tr + n_va]]
        out["test"] += [rows[i] for i in idx[n_tr + n_va:]]
    return out[split]


for _s in range(3):
    sources.EXTRA[f"busirand{_s}"] = (lambda split, s=_s: _busirand(split, s), sources._busi_load)
BREAST = common.ROOT / "external" / "BrEaST"


def _breast(split):
    """BrEaST (Pawlowska et al., Sci Data 2024; TCIA BrEaST-Lesions-USG, Dec-15-2023 release):
    256 patients x 1 image, 154 benign / 98 malignant / 4 normal. GT = *_tumor.png only (_other masks ignored).
    External-test only; never used for any selection."""
    if split != "all":
        raise ValueError("BrEaST is external-test only")
    import pandas as pd
    df = pd.read_excel(BREAST / "clinical.xlsx")
    return [dict(_src="breast", category=str(r.Classification).lower(), image_file=r.Image_filename,
                 mask_file=r.Mask_tumor_filename if isinstance(r.Mask_tumor_filename, str) else None,
                 patient=f"breast{int(r.CaseID)}") for r in df.itertuples()]


def _breast_load(row):
    d = BREAST / "data" / "BrEaST-Lesions_USG-images_and_masks"
    gray = _read(d / row["image_file"])
    mask = np.zeros_like(gray, np.uint8)
    if row["mask_file"]:
        m = _read(d / row["mask_file"])
        mask = (cv2.resize(m, gray.shape[::-1], interpolation=cv2.INTER_NEAREST) if m.shape != gray.shape else m) > 127
        mask = mask.astype(np.uint8)
    if (row["category"] == "normal") != (not mask.any()):
        raise ValueError(f"label/mask mismatch {row['image_file']}")
    return dict(id=f"breast/{row['image_file']}", image=Image.fromarray(gray).convert("RGB"), union=mask,
                instances=sources.components(mask), category=row["category"], patient=row["patient"], source="breast")


sources.EXTRA["breast"] = (_breast, _breast_load)
sources.EXTRA["busbra"] = (_busbra, _busbra_load)
sources.EXTRA["busuc"] = (_busuc, _busuc_load)
