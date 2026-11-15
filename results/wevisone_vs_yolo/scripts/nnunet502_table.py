"""nnU-Net v2 2d trained on BUS-BRA only (Dataset502, CMT-BUSNet-style setting), raw operating point.
BUS-BRA has no normal images, so no presence gate can be calibrated; FPR on BUSI/UCLM normals is raw.
Writes runs/comparison/NNUNET502.md + nnunet502.json."""
from __future__ import annotations

import json

import analyze as A
from f1_table import counts, f1

RUN = A.RUNS / "nnunet" / "busbra_s0"
SETS = [("busbra:val", "BUS-BRA val (in-domain)"), ("busbra:test", "BUS-BRA test (in-domain)"),
        ("busirand0:train+busirand0:val+busirand0:test", "BUSI uncleaned lesions 647 (external)"), ("uclm:all", "UCLM all (external)")]


def main():
    out, md = {}, ["# nnU-Net trained on BUS-BRA only (raw, no gate)", "",
                   "| Test set | n | normals | FPR % | lesion Dice | miss % | image-level F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for spec, name in SETS:
        recs = A.wevis_records(RUN, spec)
        les = [r for r in recs if r["category"] != "normal"]
        nor = [r for r in recs if r["category"] == "normal"]
        fpr = 100 * sum(r["pred_pixels"] > 0 for r in nor) / len(nor) if nor else None
        d = dict(n=len(recs), normals=len(nor), fpr=fpr, dice=sum(r["dice"] for r in les) / len(les),
                 miss=100 * sum(r["intersection"] == 0 for r in les) / len(les), f1=f1(*counts(recs))[2])
        out[spec] = d
        md.append(f"| {name} | {d['n']} | {d['normals']} | {'–' if fpr is None else f'{fpr:.1f}'} | {d['dice']:.3f} | "
                  f"{d['miss']:.1f} | {d['f1']:.3f} |")
    (A.OUT / "nnunet502.json").write_text(json.dumps(out, indent=1))
    (A.OUT / "NNUNET502.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
