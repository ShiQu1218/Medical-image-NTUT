"""How fragile is 'threshold = max over val normals'? Cached scores only, no inference.

Rules, all fitted on the model's own BUSI val split (13 normals / 72 lesions), never on test:
  max      : max val-normal score (current gate / cal)
  q90/q80  : 90th / 80th percentile of val-normal scores (accept ~10% / ~20% val FPR)
  youden   : threshold maximising sensitivity + specificity on val (normals vs lesions)
Stability: 2000 bootstrap resamples of the 13 val normals -> threshold of the 'max' rule -> test FPR / miss
spread (5th-95th percentile), i.e. how much the operating point moves with which normals happen to be in val.
Models: WeVisOne baseline / vitfull / ms_bra / ms_bra_vitfull _s42/43/44 (presence score) and YOLO26m busi_m_s42/43/44 (max det conf).
Writes runs/comparison/GATE_RULES.md + gate_rules.json
"""
from __future__ import annotations

import json

import numpy as np

import analyze as A
from presence_experiment import gated

SPECS = ["busi:test", "uclm:test", "uclm:all", "busbra:test", "breast:all"]
WEVIS_ARMS = [("baseline", "WeVisOne gate (LLM LoRA)"), ("vitfull", "WeVisOne full-ViT fine-tune"),
              ("ms_bra", "BUSI+BRA WeVisOne LLM LoRA"), ("ms_bra_vitfull", "BUSI+BRA WeVisOne full-ViT fine-tune")]
RNG = np.random.default_rng(2026)


def youden(val):
    s = np.array([r["presence_score"] for r in val], float)
    y = np.array([r["category"] != "normal" for r in val])
    best, thr = -1, None
    for t in np.unique(s):
        j = (s[y] > t).mean() + (s[~y] <= t).mean() - 1
        if j > best:
            best, thr = j, float(t)
    return thr


def rules(val):
    neg = np.array([r["presence_score"] for r in val if r["category"] == "normal"], float)
    return {"max": float(neg.max()), "q90": float(np.quantile(neg, 0.9)), "q80": float(np.quantile(neg, 0.8)),
            "youden": youden(val)}, neg


def model(kind, seed):
    if kind != "yolo":
        run = A.WEVIS_DIR / f"{kind}_s{seed}"
        val = A.wevis_records(run, A.wevis_val_spec(run))
        apply = {spec: (lambda recs: (lambda t: gated(recs, t)))(A.wevis_records(run, spec)) for spec in SPECS}
    else:
        run = A.YOLO_DIR / f"busi_m_s{seed}"
        vrec, vdet = A.yolo_dets(run, A.yolo_val_spec(run))
        val = A.dets_to_records(vrec, vdet, 1.0)
        apply = {}
        for spec in SPECS:
            recs, dets = A.yolo_dets(run, spec)
            apply[spec] = (lambda recs, dets: (lambda t: A.dets_to_records(recs, dets, t)))(recs, dets)
    return val, apply


def main():
    out, md = {}, [__doc__.split("\n", 1)[0], "", __doc__.split("\n", 1)[1].strip(), ""]
    for kind, label in WEVIS_ARMS + [("yolo", "YOLO26m")]:
        per = {}
        for s in A.SEEDS:
            val, apply = model(kind, s)
            thr, neg = rules(val)
            boot_thr = [float(RNG.choice(neg, len(neg)).max()) for _ in range(2000)]
            for spec in SPECS:
                f = apply[spec]
                for name, t in thr.items():
                    m = A.summarize(f(t))
                    per.setdefault((spec, name), []).append((m["fpr"], m["lesion_dice"], m["zero_overlap"]))
                # stability of the max rule: evaluate on the distinct bootstrap thresholds only
                cache = {t: A.summarize(f(t)) for t in set(boot_thr)}
                bs = np.array([(cache[t]["fpr"] if cache[t]["fpr"] is not None else np.nan, cache[t]["zero_overlap"])
                               for t in boot_thr], float)
                per.setdefault((spec, "boot"), []).append(bs)
            per.setdefault(("_thr", "_"), []).append(thr)
        res = {}
        for spec in SPECS:
            res[spec] = {}
            for name in ("max", "q90", "q80", "youden"):
                v = np.array(per[(spec, name)], float)
                res[spec][name] = dict(fpr=A.mean_sd(list(v[:, 0])) if not np.isnan(v[:, 0]).all() else None,
                                       lesion_dice=A.mean_sd(list(v[:, 1])), miss=A.mean_sd(list(v[:, 2])))
            b = np.concatenate(per[(spec, "boot")])
            res[spec]["max_bootstrap_p5_p95"] = dict(fpr=[float(x) for x in np.nanpercentile(b[:, 0], [5, 95])] if not np.isnan(b[:, 0]).all() else None,
                                                     miss=[float(x) for x in np.percentile(b[:, 1], [5, 95])])
        res["thresholds"] = per[("_thr", "_")]
        out[kind] = res
        md += [f"## {label}", "", "Thresholds per seed: " + "; ".join(
            ", ".join(f"{k} {v:.3g}" for k, v in t.items()) for t in res["thresholds"]), ""]
        md += ["| Set | rule | FPR % | lesion Dice | miss % |", "|---|---|---:|---:|---:|"]
        for spec in SPECS:
            for name in ("max", "q90", "q80", "youden"):
                r = res[spec][name]
                md.append(f"| {spec} | {name} | {A.fmt(r['fpr'], True) if r['fpr'] else '–'} | {A.fmt(r['lesion_dice'])} | {A.fmt(r['miss'], True)} |")
            bb = res[spec]["max_bootstrap_p5_p95"]
            fp = f"{100 * bb['fpr'][0]:.1f}–{100 * bb['fpr'][1]:.1f}" if bb["fpr"] else "–"
            md.append(f"| {spec} | max, val-normal bootstrap 5–95% | {fp} | | {100 * bb['miss'][0]:.1f}–{100 * bb['miss'][1]:.1f} |")
        md += ["", "Per seed (rule: FPR / lesion Dice / miss %):", "", "| Set | rule | " + " | ".join(f"s{s}" for s in A.SEEDS) + " |",
               "|---|---|" + "---:|" * len(A.SEEDS)]
        for spec in SPECS:
            for name in ("max", "q90", "q80", "youden"):
                cells = [("–" if f is None or f != f else f"{100 * f:.1f}") + f" / {d:.3f} / {100 * z:.1f}" for f, d, z in per[(spec, name)]]
                md.append(f"| {spec} | {name} | " + " | ".join(cells) + " |")
        md.append("")
    (A.OUT / "gate_rules.json").write_text(json.dumps(out, indent=2, default=float))
    (A.OUT / "GATE_RULES.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
