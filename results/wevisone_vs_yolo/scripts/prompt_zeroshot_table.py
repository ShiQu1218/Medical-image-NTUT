"""Zero-shot vs BUSI-fine-tuned WeVisOne, original vs '+None' prompt, on every test set (k/N counts).

Writes runs/comparison/PROMPT_ZEROSHOT.md and prompt_zeroshot.json.
"""
from __future__ import annotations

import json

import analyze as A

RUNS = {"zero-shot": ["zeroshot"],
        "fine-tuned, original prompt": [f"baseline_s{s}" for s in (42, 43, 44)],
        "fine-tuned, +None prompt": [f"prompt_none_s{s}" for s in (42, 43, 44)]}


def main():
    out, md = {}, ["# Zero-shot vs fine-tuned WeVisOne, original vs '+None' prompt",
                   "",
                   "k/N summed over seeds. FPR = normal images with any predicted pixel; zero-overlap = lesion images whose "
                   "prediction does not touch the GT; gate = presence threshold from busi:val normals. The '+None' prompt was "
                   "proposed after looking at UCLM errors, so its UCLM numbers are exploratory.",
                   "",
                   "| Set | Model | op | FPR (k/N) | lesion Dice (mean ± SD over seeds) | zero-overlap (k/N) |",
                   "|---|---|---|---:|---:|---:|"]
    for spec in A.TESTS:
        for name, runs in RUNS.items():
            per = []
            for r in runs:
                got = A.model_ops("wevis", A.RUNS / "wevis" / r, spec)
                if got:
                    per.append({op: A.summarize(recs) for op, recs in got[0].items()})
            if not per:
                continue
            for op in ("raw", "gate"):
                rows = [p[op] for p in per if op in p]
                if not rows:
                    continue
                fp, nn = sum(x["fp_k"] for x in rows), sum(x["normal_n"] for x in rows)
                mk, ln = sum(x["miss_k"] for x in rows), sum(x["lesion_n"] for x in rows)
                d = A.mean_sd([x["lesion_dice"] for x in rows])
                out.setdefault(spec, {}).setdefault(name, {})[op] = dict(fp_k=fp, normal_n=nn, miss_k=mk, lesion_n=ln, lesion_dice=d)
                fpr = f"{100 * fp / nn:.1f}% ({fp}/{nn})" if nn else "–"
                md.append(f"| {spec} | {name} (n={len(rows)}) | {op} | {fpr} | {d[0]:.3f} ± {d[1]:.3f} | "
                          f"{100 * mk / ln:.1f}% ({mk}/{ln}) |")
    (A.OUT / "prompt_zeroshot.json").write_text(json.dumps(out, indent=1))
    (A.OUT / "PROMPT_ZEROSHOT.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", A.OUT / "PROMPT_ZEROSHOT.md")


if __name__ == "__main__":
    main()
