"""Author-curated duplicate groups plus exact-pixel duplicate checks."""
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

ERRATA_SHA256 = "e3c2abc0ce027401dc717bf8a7ec76442e30c3cda2df32a0bc8aa197ff1c74a4"


def read_gray(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def load_records(root, errata, split_seed=2026):
    if hashlib.sha256(errata.read_bytes()).hexdigest() != ERRATA_SHA256:
        raise ValueError("Unexpected errata CSV; verify provenance")
    with errata.open(encoding="utf-8-sig", newline="") as stream:
        groups = list(csv.DictReader(stream, delimiter=";"))
    records = []
    parent = list(range(len(groups)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    hashes = {}
    seen = set()
    for group, row in enumerate(groups):
        for name in row["Filename"].strip("{}").split("&"):
            category = name.split(" (")[0]
            relative = f"{category}/{name}.png"
            if relative in seen:
                raise ValueError(f"Repeated errata entry: {relative}")
            seen.add(relative)
            image = read_gray(root / relative)
            digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if digest in hashes:
                parent[find(group)] = find(hashes[digest])
            hashes[digest] = group
            masks = sorted((root / category).glob(f"{name}_mask*.png"))
            if not masks:
                raise ValueError(f"Missing mask: {relative}")
            for mask in masks:
                if read_gray(mask).shape != image.shape:
                    raise ValueError(f"Mask/image shape mismatch: {mask}")
            records.append(dict(image=relative, category=category, author_group=group,
                                objection=row["Objection"], annotation=row["Annotation"],
                                sha256_pixels=digest, masks=[p.relative_to(root).as_posix() for p in masks]))
    actual = {p.relative_to(root).as_posix() for p in root.glob("*/*.png") if "_mask" not in p.stem}
    if actual != seen or len(records) != 780:
        raise ValueError("BUSI files and the 780-entry author manifest do not match")
    merged = {}
    for row in records:
        row["group"] = f"g{find(row['author_group']):04d}"
        merged.setdefault(row["group"], []).append(row)
    strata = {}
    for group, members in merged.items():
        categories = {r["category"] for r in members}
        objections = sorted({r["objection"] for r in members if r["objection"]})
        if len(categories) > 1 and "multiclass" not in objections:
            objections.append("multiclass")
        # Keep the first filename from the published group, without inspecting masks or scores.
        for index, row in enumerate(members):
            row["exclusion"] = "&".join(objections) if objections else ("duplicate" if index else "")
            row["clean"] = not row["exclusion"]
        stratum = "+".join(sorted(categories))
        strata.setdefault(stratum, []).append(group)
    rng = np.random.default_rng(split_seed)
    for names in strata.values():
        names = sorted(names)
        rng.shuffle(names)
        ntrain, nval = round(len(names)*0.7), round(len(names)*0.15)
        for index, name in enumerate(names):
            split = "train" if index < ntrain else ("val" if index < ntrain+nval else "test")
            for row in merged[name]:
                row["split"] = split
    return records


def load_arrays(root, records, size):
    images, masks = [], []
    for row in records:
        image = read_gray(root / row["image"])
        mask = np.zeros_like(image, dtype=np.uint8)
        for relative in row["masks"]:
            mask |= (read_gray(root / relative) > 0).astype(np.uint8)
        if row["category"] == "normal" and mask.any():
            raise ValueError(f"Normal case has a nonempty mask: {row['image']}")
        if row["category"] != "normal" and not mask.any():
            raise ValueError(f"Lesion case has an empty mask: {row['image']}")
        images.append(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA))
        masks.append(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST))
    return np.stack(images), np.stack(masks)
