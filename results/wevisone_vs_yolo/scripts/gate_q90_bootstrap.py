"""Paired patient-cluster bootstrap with the q90 gate (90th percentile of the 13 BUSI val-normal presence scores).

Rule fixed on val only, applied identically to every arm. Cached scores, no inference.
Comparisons: full-ViT vs LLM-only LoRA (BUSI-trained and BUSI+BRA-trained), both q90; plus full-ViT q90 vs the
current main method (LLM-only LoRA, max rule). Holm over every p-value in this table (FPR and lesion Dice).
Writes runs/comparison/GATE_Q90_BOOTSTRAP.md + gate_q90_bootstrap.json
"""
from __future__ import annotations

import json

import analyze as A
from gate_rules import rules
from presence_experiment import gated

SPECS = ["busi:test", "uclm:test", "uclm:all", "busuc:all", "busbra:test", "breast:all"]
PAIRS = [("vitfull", "q90", "baseline", "q90", "full-ViT q90 vs LLM-LoRA q90"),
         ("ms_bra_vitfull", "q90", "ms_bra", "q90", "BUSI+BRA full-ViT q90 vs BUSI+BRA LLM-LoRA q90"),
         ("vitfull", "q90", "baseline", "max", "full-ViT q90 vs LLM-LoRA max (current main)")]


def arm(name, rule):
    out = {}
    for s in A.SEEDS:
        run = A.WEVIS_DIR / f"{name}_s{s}"
        t = rules(A.wevis_records(run, A.wevis_val_spec(run)))[0][rule]
        for spec in SPECS:
            out.setdefault(spec, []).append(gated(A.wevis_records(run, spec), t))
    return out


def main():
    cache, boot, desc = {}, {}, {}
    for a, ra, b, rb, label in PAIRS:
        for key in ((a, ra), (b, rb)):
            if key not in cache:
                cache[key] = arm(*key)
                for spec in SPECS:
                    ms = [A.summarize(r) for r in cache[key][spec]]
                    desc[(key, spec)] = {m: A.mean_sd([x[m] for x in ms]) for m in ("fpr", "lesion_dice", "zero_overlap")
                                         if ms[0][m] is not None}
        for spec in SPECS:
            boot[f"{spec} | {label}"] = A.paired_bootstrap(cache[(a, ra)][spec], cache[(b, rb)][spec])
    ps = {}
    for k, v in boot.items():
        for m in ("fpr", "lesion_dice"):
            if v[f"{m}_p"] == v[f"{m}_p"]:
                ps[(k, m)] = v[f"{m}_p"]
    adj = A.holm(ps)
    md = [__doc__.split("\n", 1)[0], "", __doc__.split("\n", 1)[1].strip(), "",
          "## Descriptive (mean ± SD over seeds 42/43/44)", "", "| Set | arm | rule | FPR % | lesion Dice | miss % |",
          "|---|---|---|---:|---:|---:|"]
    for spec in SPECS:
        for (name, rule) in cache:
            d = desc[((name, rule), spec)]
            md.append(f"| {spec} | {name} | {rule} | {A.fmt(d['fpr'], True) if 'fpr' in d else '–'} | "
                      f"{A.fmt(d['lesion_dice'])} | {A.fmt(d['zero_overlap'], True)} |")
    md += ["", "## Paired bootstrap (A − B; 5000 patient-cluster resamples; Holm over this table)", "",
           "| Comparison | images / patients | Δ FPR pp [95% CI] | Holm p | Δ lesion Dice [95% CI] | Holm p |",
           "|---|---|---:|---:|---:|---:|"]
    for k, v in boot.items():
        v["fpr_p_holm"] = adj.get((k, "fpr"), float("nan"))
        v["lesion_dice_p_holm"] = adj.get((k, "lesion_dice"), float("nan"))
        f = (f"{100 * v['fpr_diff']:+.1f} [{100 * v['fpr_diff_ci'][0]:+.1f}, {100 * v['fpr_diff_ci'][1]:+.1f}] | "
             f"{v['fpr_p_holm']:.2g}") if v["fpr_diff"] == v["fpr_diff"] else "– | –"
        md.append(f"| {k.replace('|', '/')} | {v['n_images']} / {v['n_patients']} | {f} | {v['lesion_dice_diff']:+.3f} "
                  f"[{v['lesion_dice_diff_ci'][0]:+.3f}, {v['lesion_dice_diff_ci'][1]:+.3f}] | {v['lesion_dice_p_holm']:.2g} |")
    (A.OUT / "gate_q90_bootstrap.json").write_text(json.dumps(boot, indent=2, default=float))
    (A.OUT / "GATE_Q90_BOOTSTRAP.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
