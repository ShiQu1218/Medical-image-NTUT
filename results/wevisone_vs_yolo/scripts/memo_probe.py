"""Memorization probe: how much do zero-shot / fine-tuned WeVisOne (and nnU-Net as a never-pretrained reference)
lose under mild label-preserving geometry (hflip, rot +-8 deg, zoom-in x1.15, shift 8%), per cohort?

If WeVisOne had memorized a cohort (e.g. BUS_UC) during pretraining, its zero-shot drop there should be clearly
larger than on the other cohorts. Reports lesion Dice and image-level localized F1 (f1_table.py definition) on the
clean and each transformed set, and the relative drop % = 100 * (clean - transformed) / clean.
Paired Dice drop uses only images that are lesion images in both versions. Writes runs/comparison/MEMO.md + memo.json.
"""
from __future__ import annotations

import json

import analyze as A
from f1_table import counts, f1

SETS = ["busi:test", "uclm:all", "busuc:all", "busbra:test", "breast:all"]
TRANSFORMS = ["hflip1", "rot1", "zoom1", "shift1"]
MODELS = [("WeVisOne zero-shot", "wevis", A.RUNS / "wevis" / "zeroshot", A.RUNS / "wevis" / "zeroshot_memo"),
          ("WeVisOne fine-tuned s42", "wevis", A.WEVIS_DIR / "baseline_s42", A.WEVIS_DIR / "baseline_s42"),
          ("nnU-Net (BUSI only)", "wevis", A.RUNS / "nnunet" / "busi_s0", A.RUNS / "nnunet" / "busi_s0")]


def recs(kind, run, spec):
    got = A.model_ops(kind, run, spec)
    return got[0]["raw"] if got else None


def stats(rs):
    les = [r for r in rs if r["category"] != "normal"]
    return dict(dice=sum(r["dice"] for r in les) / len(les), f1=f1(*counts(rs))[2], n=len(rs))


def pct(clean, x):
    return 100 * (clean - x) / clean if clean else float("nan")


def main():
    out, md = {}, ["# Memorization probe (raw operating point, seed 42 / single model)", "",
                   __doc__.split("\n\n")[1].replace("\n", " "), "",
                   "| Model | Set | clean Dice | clean F1 | " + " | ".join(f"{t} Dice drop % | {t} F1 drop %" for t in TRANSFORMS)
                   + " | mean Dice drop % | mean F1 drop % | paired Dice drop (abs, mean) |",
                   "|---|---|---:|---:|" + "---:|---:|" * len(TRANSFORMS) + "---:|---:|---:|"]
    for name, kind, clean_run, probe_run in MODELS:
        for spec in SETS:
            c = recs(kind, clean_run, spec)
            if not c:
                continue
            cs, cells, dd, fd, paired = stats(c), [], [], [], []
            cdice = {r["id"]: r["dice"] for r in c if r["category"] != "normal"}
            for t in TRANSFORMS:
                x = recs(kind, probe_run, f"{spec}~{t}")
                if not x:
                    cells.append("– | –")
                    continue
                xs = stats(x)
                d, f = pct(cs["dice"], xs["dice"]), pct(cs["f1"], xs["f1"])
                dd.append(d); fd.append(f)
                paired += [cdice[r["id"]] - r["dice"] for r in x if r["category"] != "normal" and r["id"] in cdice]
                out.setdefault(name, {}).setdefault(spec, {})[t] = dict(xs, dice_drop_pct=d, f1_drop_pct=f)
                cells.append(f"{d:+.1f} | {f:+.1f}")
            out.setdefault(name, {}).setdefault(spec, {})["clean"] = cs
            mean = lambda v: f"{sum(v) / len(v):+.1f}" if v else "–"
            md.append(f"| {name} | {spec} | {cs['dice']:.3f} | {cs['f1']:.3f} | " + " | ".join(cells)
                      + f" | **{mean(dd)}** | **{mean(fd)}** | {(sum(paired) / len(paired)) if paired else float('nan'):+.3f} |")
    (A.OUT / "memo.json").write_text(json.dumps(out, indent=1))
    (A.OUT / "MEMO.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
