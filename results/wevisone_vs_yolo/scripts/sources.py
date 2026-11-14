"""Uniform dataset access: rows(spec) -> light descriptors, load(row) -> item dict
dict(id, image(PIL RGB), union(np uint8), instances[list np uint8], category, patient, source, ...).

Specs
  busi:train|val|test            fixed cleaned split (368/85/76)
  busicv<k>:train|val|test       grouped, category-stratified 5-fold over all 529 clean BUSI images;
                                 test = fold k, val = fold (k+1)%5, train = the other three
  uclm:all                       BUS-UCLM v3 (679 usable)
  uclm:train|val|test|dev        patient-level split of UCLM (seed 2026): test ~50% of patients, dev = train+val
  busbra:..., busuc:all          see sources_extra.py
  <src>:<split>+<src>:<split>    concatenation
Any spec may end with @<fraction>/<seed> to take a group-level, category-stratified subset.
"""
from __future__ import annotations

import numpy as np

import common

UCLM_ROOT = common.ROOT / "external" / "BUS-UCLM"
K_FOLDS = 5


def components(union):
    """Instances from a union mask = connected components (datasets without instance masks)."""
    import cv2
    n, lab = cv2.connectedComponents(union.astype(np.uint8))
    return [(lab == i).astype(np.uint8) for i in range(1, n) if (lab == i).sum() >= 16]


def _stratified_groups(rows, key_group, n_bins, seed):
    """Assign each group to one of n_bins, round-robin within category after a seeded shuffle."""
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], set()).add(r[key_group])
    rng = np.random.default_rng(seed)
    assign = {}
    for cat in sorted(by_cat):
        groups = sorted(by_cat[cat])
        rng.shuffle(groups)
        offset = int(rng.integers(n_bins))
        for i, g in enumerate(groups):
            assign.setdefault(g, (i + offset) % n_bins)
    return assign


# ---------------- BUSI ----------------
def _busi_all():
    return [r for s in ("train", "val", "test") for r in common.clean_split(s)]


def _busi(split):
    return [dict(r, _src="busi") for r in common.clean_split(split)]


def _busicv(k, split):
    rows = _busi_all()
    fold = _stratified_groups(rows, "group", K_FOLDS, seed=2026)
    want = {"test": {k}, "val": {(k + 1) % K_FOLDS}, "train": set(range(K_FOLDS)) - {k, (k + 1) % K_FOLDS}}[split]
    return [dict(r, _src="busi", fold=fold[r["group"]]) for r in rows if fold[r["group"]] in want]


def _busi_load(row):
    image, instances, union = common.load_sample(row)
    return dict(id=row["image"], image=image, union=union, instances=instances, category=row["category"],
                patient=row["group"], source="busi")


# ---------------- BUS-UCLM ----------------
def _uclm(split):
    from external_test import load_uclm, sample
    rows = []
    for row in load_uclm(UCLM_ROOT):
        try:
            sample(UCLM_ROOT, row)
        except ValueError:
            continue  # same 4 explicit exclusions as external_test.py
        rows.append(dict(row, _src="uclm", category=row["Label"].lower(), patient=row["Image"].split("_")[0]))
    if split == "all":
        return rows
    # Patient-level split: 10 bins stratified by each patient's majority label.
    # test = bins 5-9 (held out), dev = bins 0-4 = train (0-3) + val (4).
    pats = {}
    for r in rows:
        pats.setdefault(r["patient"], []).append(r["category"])
    major = [dict(patient=p, category=max(set(c), key=c.count)) for p, c in pats.items()]
    assign = _stratified_groups(major, "patient", 10, seed=2026)
    want = {"test": range(5, 10), "dev": range(0, 5), "train": range(0, 4), "val": range(4, 5)}[split]
    return [r for r in rows if assign[r["patient"]] in want]


def _uclm_load(row):
    from external_test import sample
    image, union, _ = sample(UCLM_ROOT, row)
    return dict(id=row["Image"], image=image, union=union, instances=components(union), category=row["category"],
                patient=row["patient"], source="uclm",
                plain=all(row[k] == "No" for k in ("Doppler", "Marks", "Combined")))


# ---------------- dispatch ----------------
EXTRA = {}  # name -> (rows_fn(split), load_fn(row)); filled by optional dataset modules
LOADERS = {"busi": _busi_load, "uclm": _uclm_load}


def _subset(rows, fraction, seed):
    """Group-level, per-category prefix of a seeded shuffle: smaller fractions nest inside larger ones."""
    gkey = lambda r: r.get("group", r.get("patient"))  # noqa: E731
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], set()).add(gkey(r))
    rng = np.random.default_rng(seed)
    keep = set()
    for cat in sorted(by_cat):
        groups = sorted(by_cat[cat])
        rng.shuffle(groups)
        keep |= set(groups[:max(1, int(round(fraction * len(groups))))])
    return [r for r in rows if gkey(r) in keep]


def rows(spec: str) -> list[dict]:
    if "~" in spec:  # synthetic corruption, e.g. busi:test~icon2 (see corruptions.py)
        base, corr = spec.split("~")
        fam, sev = corr[:-1], int(corr[-1])
        return [dict(r, _corrupt=(fam, sev)) for r in rows(base)]
    if "+" in spec:
        return [r for part in spec.split("+") for r in rows(part)]
    sub = None
    if "@" in spec:
        spec, sub = spec.split("@")
    source, split = spec.split(":")
    if source == "busi":
        out = _busi(split)
    elif source.startswith("busicv"):
        out = _busicv(int(source[len("busicv"):]), split)
    elif source == "uclm":
        out = _uclm(split)
    elif source in EXTRA:
        out = EXTRA[source][0](split)
    else:
        raise ValueError(spec)
    if sub:
        frac, seed = sub.split("/")
        out = _subset(out, float(frac), int(seed))
    return out


def load(row: dict) -> dict:
    src = row["_src"]
    it = LOADERS[src](row) if src in LOADERS else EXTRA[src][1](row)
    if "_corrupt" in row:
        from PIL import Image
        import corruptions
        fam, sev = row["_corrupt"]
        gray, inst = corruptions.apply(np.asarray(it["image"].convert("L")), it["instances"], fam, sev, it["id"])
        union = np.zeros_like(gray, np.uint8)
        for m in inst:
            union |= m.astype(np.uint8)
        it = dict(it, image=Image.fromarray(gray).convert("RGB"), instances=inst, union=union,
                  corruption=f"{fam}{sev}", category=it["category"] if inst or it["category"] == "normal" else "normal")
    return it


def items(spec: str):
    return (load(r) for r in rows(spec))


try:  # optional extra datasets (BUS-BRA etc.), registered into EXTRA
    import sources_extra  # noqa: F401
except ImportError:
    pass
