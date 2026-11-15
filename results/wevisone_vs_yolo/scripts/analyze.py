"""Aggregate WeVisOne vs YOLO results into runs/comparison/{report.json, REPORT.md}.

Operating points (thresholds are always fixed on the model's own validation spec, never on test):
  WeVisOne raw   : generated masks as is
  WeVisOne gate  : keep masks iff presence_score > max(val normal presence_score)
  YOLO 0.25      : detections with conf >= 0.25 (Ultralytics default)
  YOLO cal       : detections with conf > max(val normal max-conf)
Threshold-free: image-level AUROC (normal vs lesion) of the presence score; sensitivity at FPR <= 10%.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import common
import sources
from fpr_metrics import metrics
from presence_experiment import gated
from yolo_eval import dets_to_records, icon_only, load_dets

RUNS = common.RUNS
OUT = RUNS / "comparison"
TESTS = ["busi:test", "uclm:test", "uclm:all", "busuc:all", "busbra:test", "breast:all"]


@lru_cache(None)
def uclm_test_ids():
    return frozenset(r["Image"] for r in sources.rows("uclm:test"))


def key(spec):
    return spec.replace(":", "_")


def _derive(spec):
    """uclm:test is evaluated as part of uclm:all and filtered by patient."""
    return ("uclm:all", lambda recs: [r for r in recs if r["id"] in uclm_test_ids()]) if spec == "uclm:test" else (spec, None)


# ---------------- loading ----------------
def wevis_records(run: Path, spec: str, mode="generate"):
    base, filt = _derive(spec)
    tag = mode if not mode.startswith("yolo") else mode
    files = sorted(run.glob(f"**/{key(base)}_{tag}.jsonl"))
    if not files:
        return None
    seen = {}
    for line in files[0].read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            bits = r.pop("pred_bits", None)
            if bits is not None:
                pred = np.unpackbits(np.frombuffer(bytes.fromhex(bits), np.uint8))[:common.EVAL_SIZE ** 2]
                r["icon_only"] = icon_only(pred.reshape(common.EVAL_SIZE, common.EVAL_SIZE).astype(bool))
            seen[r["id"]] = r
    recs = list(seen.values())
    return filt(recs) if filt else recs


def wevis_val_spec(run: Path):
    a = run / "args.json"
    return json.loads(a.read_text()).get("val_spec", "busi:val") if a.exists() else "busi:val"


def yolo_dets(run: Path, spec: str):
    base, filt = _derive(spec)
    files = sorted(run.glob(f"eval*/{key(base)}_dets.npz"))
    if not files:
        return None
    recs, dets = load_dets(files[0])
    return (filt(recs) if filt else recs), dets


def yolo_val_spec(run: Path):
    return json.loads((run / "eval" / "report.json").read_text())["calibration_split"]


# ---------------- metrics ----------------
def auroc(recs):
    pos = np.array([r["presence_score"] for r in recs if r["category"] != "normal"], float)
    neg = np.array([r["presence_score"] for r in recs if r["category"] == "normal"], float)
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()
    return float(gt)


def sens_at_fpr(recs, fpr=0.10):
    neg = np.sort([r["presence_score"] for r in recs if r["category"] == "normal"])[::-1]
    pos = np.array([r["presence_score"] for r in recs if r["category"] != "normal"])
    if not len(neg) or not len(pos):
        return None
    k = int(np.floor(fpr * len(neg)))  # allow k false positives
    thr = neg[k] if k < len(neg) else -np.inf
    return float((pos > thr).mean())


def summarize(recs):
    m = metrics(recs)
    normals = [r for r in recs if r["category"] == "normal"]
    fpr_no_icon = (sum(r["pred_pixels"] > 0 and not r.get("icon_only", False) for r in normals) / len(normals)) if normals else None
    return dict(n=m["n"], normal_n=m["normal_n"], lesion_n=m["lesion_n"], fpr=m["normal_fp_rate"], fpr_no_icon=fpr_no_icon,
                fp_k=m["fp"], miss_k=sum(r["intersection"] == 0 for r in recs if r["category"] != "normal"),
                lesion_dice=m["lesion_dice"], dice=m["dice"], zero_overlap=m["lesion_zero_overlap_rate"],
                empty=m["lesion_empty_rate"])


def model_ops(kind: str, run: Path, spec: str):
    """Returns {op_name: records} plus threshold-free stats for one model run on one spec."""
    if kind == "wevis":
        recs = wevis_records(run, spec)
        if recs is None:
            return None
        val = wevis_records(run, wevis_val_spec(run))
        thr = max([r["presence_score"] for r in val if r["category"] == "normal"], default=None) if val else None
        ops = {"raw": recs}
        if thr is not None:
            ops["gate"] = gated(recs, thr)
        return ops, recs
    got = yolo_dets(run, spec)
    if got is None:
        return None
    recs, dets = got
    vrec, vdet = yolo_dets(run, yolo_val_spec(run))
    ops, scored = {}, None
    for suffix, drop in (("", False), ("_noicon", True)):
        thr = max([r["presence_score"] for r in dets_to_records(vrec, vdet, 1.0, drop_icon=drop)
                   if r["category"] == "normal"], default=0.0)
        raw = dets_to_records(recs, dets, 0.25, strict=False, drop_icon=drop)
        ops["0.25" + suffix], ops["cal" + suffix] = raw, dets_to_records(recs, dets, thr, drop_icon=drop)
        if not drop:
            scored = raw
        else:
            ops["_scored_noicon"] = raw
    return ops, scored


def evaluate_model(kind, run, specs):
    out = {}
    for spec in specs:
        got = model_ops(kind, run, spec)
        if got is None:
            continue
        ops, scored = got
        ops = dict(ops)
        alt = ops.pop("_scored_noicon", scored)
        out[spec] = dict(auroc=auroc(scored), auroc_noicon=auroc(alt), sens_at_fpr10=sens_at_fpr(scored),
                         sens_at_fpr10_noicon=sens_at_fpr(alt), **{op: summarize(r) for op, r in ops.items()})
    return out


def mean_sd(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)]


def aggregate(per_seed: list[dict]):
    """per_seed: list of evaluate_model outputs -> same structure with [mean, sd, n] leaves."""
    out = {}
    specs = set().union(*[p.keys() for p in per_seed]) if per_seed else set()
    for spec in specs:
        ps = [p[spec] for p in per_seed if spec in p]
        out[spec] = {}
        for k in ps[0]:
            if isinstance(ps[0][k], dict):
                out[spec][k] = {m: (sum(p[k][m] for p in ps if k in p) if m in ("fp_k", "miss_k", "normal_n", "lesion_n", "n")
                                    else mean_sd([p[k][m] for p in ps if k in p])) for m in ps[0][k]}
            else:
                out[spec][k] = mean_sd([p[k] for p in ps])
    return out


# ---------------- paired patient-cluster bootstrap ----------------
def per_image(recs):
    return {r["id"]: (r["category"], r.get("patient"), float(r["pred_pixels"] > 0), r["dice"]) for r in recs}


def paired_bootstrap(a_seeds, b_seeds, n_boot=5000, seed=2026):
    """a_seeds/b_seeds: lists of record lists (one per seed). Per image, average over seeds, then
    resample patients. Returns diff (A - B) for FPR and lesion Dice with 95% CI."""
    A = [per_image(r) for r in a_seeds]
    B = [per_image(r) for r in b_seeds]
    ids = sorted(set.intersection(*[set(x) for x in A + B]))
    info = {i: A[0][i][:2] for i in ids}
    fp = {i: np.mean([x[i][2] for x in A]) - np.mean([x[i][2] for x in B]) for i in ids}
    dc = {i: np.mean([x[i][3] for x in A]) - np.mean([x[i][3] for x in B]) for i in ids}
    patients = sorted({info[i][1] for i in ids})
    by_p = {p: [i for i in ids if info[i][1] == p] for p in patients}
    rng = np.random.default_rng(seed)

    def stat(sample):
        norm = [i for p in sample for i in by_p[p] if info[i][0] == "normal"]
        les = [i for p in sample for i in by_p[p] if info[i][0] != "normal"]
        return (np.mean([fp[i] for i in norm]) if norm else np.nan, np.mean([dc[i] for i in les]) if les else np.nan)

    point = stat(patients)
    boots = np.array([stat(rng.choice(patients, len(patients))) for _ in range(n_boot)])
    ci = lambda col: [float(np.nanpercentile(boots[:, col], 2.5)), float(np.nanpercentile(boots[:, col], 97.5))]  # noqa: E731
    return dict(n_images=len(ids), n_patients=len(patients),
                fpr_diff=float(point[0]), fpr_diff_ci=ci(0), lesion_dice_diff=float(point[1]), lesion_dice_diff_ci=ci(1))


def operating_curve(kind, run, spec):
    """(FPR, lesion Dice, zero-overlap rate) over every threshold of the image-level score.
    WeVisOne/nnU-Net: gate on presence_score; YOLO: detection confidence threshold."""
    if kind == "yolo":
        got = yolo_dets(run, spec)
        if not got:
            return None
        recs, dets = got
        # Per image: detections sorted by conf; cumulative union -> (pred>0, inter, dice) for keeping top-k.
        pre = []
        for r in recs:
            d = dets[r["id"]]
            order = np.argsort(-d["conf"])
            conf = d["conf"][order]
            masks = np.unpackbits(d["masks"], axis=-1)[:, :, :common.EVAL_SIZE].astype(bool)[order]
            gt = np.unpackbits(d["gt"], axis=-1)[:, :common.EVAL_SIZE].astype(bool)
            u = np.zeros_like(gt)
            stats = [(False, 0, common.scores(u, gt)[0])]
            for m in masks:
                u = u | m
                stats.append((bool(u.any()), int((u & gt).sum()), common.scores(u, gt)[0]))
            pre.append((r["category"] == "normal", conf, stats))
        thrs = sorted({0.0} | {float(c) for _, conf, _ in pre for c in conf})
        step = max(1, len(thrs) // 400)
        pts = []
        for t in thrs[::step] + [thrs[-1]]:
            fp = nn_ = 0
            ld, miss = [], 0
            for is_norm, conf, stats in pre:
                k = int((conf > t).sum())
                pos, inter, dice = stats[k]
                if is_norm:
                    nn_ += 1
                    fp += pos
                else:
                    ld.append(dice)
                    miss += inter == 0
            pts.append((fp / nn_ if nn_ else None, float(np.mean(ld)) if ld else None, miss / len(ld) if ld else None, t))
        return pts
    else:
        recs = wevis_records(run, spec)
        if not recs:
            return None
        thrs = sorted({-1e9} | {float(r["presence_score"]) for r in recs})
        fn = lambda t: gated(recs, t)  # noqa: E731
    pts = []
    step = max(1, len(thrs) // 400)
    for t in thrs[::step] + [thrs[-1]]:
        s = summarize(fn(t))
        pts.append((s["fpr"], s["lesion_dice"], s["zero_overlap"], t))
    return pts


def at_fpr(pts, target):
    """Best lesion Dice among thresholds with FPR <= target (thresholds picked on the test set: analysis only)."""
    ok = [p for p in pts if p[0] is not None and p[0] <= target + 1e-9]
    return max(ok, key=lambda p: p[1]) if ok else None


def op_records(kind, run, spec, op):
    got = model_ops(kind, run, spec)
    return None if got is None else got[0].get(op)


# ---------------- report ----------------
YOLO_DIR, WEVIS_DIR = RUNS / "yolo", RUNS / "wevis"
SEEDS = (42, 43, 44)


def _dice_pair(rows, dets):
    """(per-image mean Dice, pooled Dice over all pixels) for lesion rows."""
    rows = [r for r in rows if r["category"] != "normal"]
    if not rows:
        return (None, None)
    gt = sum(int(np.unpackbits(dets[r["id"]]["gt"]).sum()) for r in rows)
    inter = sum(r["intersection"] for r in rows)
    pp = sum(r["pred_pixels"] for r in rows)
    return (float(np.mean([r["dice"] for r in rows])), float(2 * inter / max(1, pp + gt)))


def hybrid_records(seed, spec, which):
    rep = json.loads((YOLO_DIR / f"busi_m_s{seed}" / "eval" / "report.json").read_text())
    thr = rep["calibrated_threshold"] if which == "cal" else 0.25
    return wevis_records(WEVIS_DIR / f"baseline_s{seed}", spec, mode=f"yolo{thr:g}")


def group(kind, runs, specs):
    per = [evaluate_model(kind, r, specs) for r in runs if r.exists()]
    per = [p for p in per if p]
    return dict(n_runs=len(per), metrics=aggregate(per)) if per else None


def oof(kind, runs_by_fold):
    """Concatenate each fold's test predictions (thresholds from each fold's own val)."""
    ops_all, scored_all = {}, []
    for k, run in runs_by_fold.items():
        got = model_ops(kind, run, f"busicv{k}:test")
        if got is None:
            return None
        ops, scored = got
        for op, recs in ops.items():
            if not op.startswith("_"):
                ops_all.setdefault(op, []).extend(recs)
        scored_all += scored
    return dict(auroc=auroc(scored_all), sens_at_fpr10=sens_at_fpr(scored_all),
                **{op: summarize(r) for op, r in ops_all.items()}), ops_all


def fmt(v, pct=False):
    if v is None:
        return "–"
    m, s, n = v
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f(m) + (f" ± {f(s)}" if n > 1 else "")


def table(title, rows, specs):
    """rows: list of (label, group_metrics, op)."""
    lines = [f"### {title}", "", "| Model | " + " | ".join(
        f"{s} FPR % | {s} FPR excl. icon % | {s} lesion Dice | {s} miss % | {s} AUROC" for s in specs) + " |",
        "|---|" + "---:|" * (5 * len(specs))]
    for label, g, op in rows:
        if not g:
            continue
        cells = []
        for s in specs:
            m = g["metrics"].get(s)
            if not m or op not in m:
                cells += ["–"] * 5
                continue
            o = m[op]
            cnt = lambda k, n: f" ({o[k]}/{o[n]})" if o.get(n) else ""  # noqa: E731  counts summed over seeds
            cells += [fmt(o["fpr"], True) + cnt("fp_k", "normal_n"), fmt(o["fpr_no_icon"], True), fmt(o["lesion_dice"]),
                      fmt(o["zero_overlap"], True) + cnt("miss_k", "lesion_n"),
                      fmt(m["auroc_noicon" if op.endswith("noicon") else "auroc"])]
        lines.append(f"| {label} (n={g['n_runs']}) | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep, md = {}, ["# WeVisOne-VL-4B vs YOLO26m-seg", "",
                   "FPR = normal images with any predicted pixel. miss = lesion images with zero GT overlap. "
                   "AUROC = image-level normal vs lesion from the presence score (WeVisOne logit gap, YOLO max conf). "
                   "Thresholds come from each model's own validation split only. mean ± SD over seeds.", ""]

    # 1) BUSI-trained models on all test sets
    g = dict(wevis=group("wevis", [WEVIS_DIR / f"baseline_s{s}" for s in SEEDS], TESTS),
             wevis_aug=group("wevis", [WEVIS_DIR / f"aug_s{s}" for s in SEEDS], TESTS),
             yolo=group("yolo", [YOLO_DIR / f"busi_m_s{s}" for s in SEEDS], TESTS))
    hyb = {}
    for which in ("cal", "0.25"):
        per = []
        for s in SEEDS:
            out = {}
            for spec in TESTS:
                recs = hybrid_records(s, spec, which) if (YOLO_DIR / f"busi_m_s{s}" / "eval" / "report.json").exists() else None
                if recs:
                    out[spec] = dict(auroc=auroc(recs), sens_at_fpr10=sens_at_fpr(recs), raw=summarize(recs))
            if out:
                per.append(out)
        hyb[which] = dict(n_runs=len(per), metrics=aggregate(per)) if per else None
    rep["busi_trained"] = dict(g, hybrid=hyb)
    rows = [("WeVisOne raw", g["wevis"], "raw"), ("WeVisOne + gate", g["wevis"], "gate"),
            ("WeVisOne strong-aug raw", g["wevis_aug"], "raw"), ("WeVisOne strong-aug + gate", g["wevis_aug"], "gate"),
            ("YOLO conf≥0.25", g["yolo"], "0.25"), ("YOLO val-calibrated", g["yolo"], "cal"),
            ("YOLO conf≥0.25, icon dets dropped", g["yolo"], "0.25_noicon"),
            ("YOLO val-calibrated, icon dets dropped", g["yolo"], "cal_noicon"),
            ("Hybrid: YOLO cal boxes → WeVisOne decoder", hyb["cal"], "raw"),
            ("Hybrid: YOLO 0.25 boxes → WeVisOne decoder", hyb["0.25"], "raw")]
    md.append("## 1. Trained on BUSI only")
    md.append(table("In-domain and patient-held-out", rows, ["busi:test", "uclm:test", "uclm:all"]))
    md.append(table("Lesion-only external sets (no normals: FPR/AUROC undefined; BrEaST has 4 normals)", rows,
                    ["busuc:all", "busbra:test", "breast:all"]))

    # 1b) matched-FPR comparison (reviewer item 3): thresholds swept on the TEST set -> operating-curve analysis only
    mf = {}
    curves = {"WeVisOne (presence gate)": ("wevis", [WEVIS_DIR / f"baseline_s{s}" for s in SEEDS]),
              "YOLO (conf threshold)": ("yolo", [YOLO_DIR / f"busi_m_s{s}" for s in SEEDS]),
              "nnU-Net (max-prob gate)": ("wevis", [RUNS / "nnunet" / "busi_s0"])}
    for spec in ["uclm:all", "uclm:test"]:
        for label, (kind, runs) in curves.items():
            for target in (0.05, 0.10, 0.15):
                pts = [at_fpr(c, target) for c in (operating_curve(kind, r, spec) for r in runs if r.exists()) if c]
                pts = [p for p in pts if p]
                if pts:
                    mf.setdefault(spec, {}).setdefault(label, {})[target] = dict(
                        n_runs=len(pts), fpr=mean_sd([p[0] for p in pts]), lesion_dice=mean_sd([p[1] for p in pts]),
                        zero_overlap=mean_sd([p[2] for p in pts]))
    rep["matched_fpr"] = mf
    if mf:
        md += ["### Lesion performance at matched normal-image FPR (thresholds swept on the test set; analysis, not a deployable rule)", "",
               "| Set | Model | runs | FPR ≤ 5%: lesion Dice / miss % | FPR ≤ 10% | FPR ≤ 15% |", "|---|---|---:|---|---|---|"]
        for spec, d in mf.items():
            for label, t in d.items():
                cell = lambda x: f"{fmt(t[x]['lesion_dice'])} / {fmt(t[x]['zero_overlap'], True)}" if x in t else "–"  # noqa: E731
                md.append(f"| {spec} | {label} | {next(iter(t.values()))['n_runs']} | {cell(0.05)} | {cell(0.10)} | {cell(0.15)} |")
        md.append("")

    # 2) GT-box oracle: decoder vs localisation
    orc = {}
    for spec in ["busi:test", "uclm:all", "uclm:test"]:
        gt = wevis_records(WEVIS_DIR / "baseline_s42", spec, "gt_box")
        gen = wevis_records(WEVIS_DIR / "baseline_s42", spec, "generate")
        if gt and gen:
            orc[spec] = dict(gt_box=summarize(gt), generate=summarize(gen))
    rep["oracle_s42"] = orc
    if orc:
        md += ["## 2. GT-box oracle (seed 42): where is the external gap?", "",
               "| Set | generate lesion Dice | GT-box lesion Dice | generate miss % |", "|---|---:|---:|---:|"]
        md += [f"| {s} | {o['generate']['lesion_dice']:.3f} | {o['gt_box']['lesion_dice']:.3f} | {100 * o['generate']['zero_overlap']:.1f} |"
               for s, o in orc.items()]
        md.append("")

    # 3) 5-fold OOF
    cv = {}
    for kind, name in [("wevis", "cv{k}_s42"), ("yolo", "cv{k}_s42")]:
        d = WEVIS_DIR if kind == "wevis" else YOLO_DIR
        got = oof(kind, {k: d / name.format(k=k) for k in range(5)})
        if got:
            cv[kind] = got
    rep["cv_oof"] = {k: v[0] for k, v in cv.items()}
    if cv:
        md += ["## 3. Grouped 5-fold CV, out-of-fold over all 529 clean BUSI images (72 normals)", "",
               "| Model | op | FPR % | lesion Dice | miss % | AUROC | sens@FPR10% |", "|---|---|---:|---:|---:|---:|---:|"]
        for kind, (m, _) in cv.items():
            for op in [o for o in m if isinstance(m[o], dict)]:
                md.append(f"| {kind} | {op} | {100 * m[op]['fpr']:.1f} ({round(m[op]['fpr'] * m[op]['normal_n'])}/{m[op]['normal_n']}) | "
                          f"{m[op]['lesion_dice']:.3f} | {100 * m[op]['zero_overlap']:.1f} | {m['auroc']:.3f} | {m['sens_at_fpr10']:.3f} |")
        md.append("")

    # 4) low data
    fr = {}
    for f in ("0.1", "0.25", "0.5"):
        fr[f] = dict(wevis=group("wevis", [WEVIS_DIR / f"frac{f}_s42"], ["busi:test", "uclm:all"]),
                     yolo=group("yolo", [YOLO_DIR / f"frac{f}_sub{s}_s{s}" for s in SEEDS], ["busi:test", "uclm:all"]))
    fr["1.0"] = dict(wevis=group("wevis", [WEVIS_DIR / f"baseline_s{s}" for s in SEEDS], ["busi:test", "uclm:all"]),
                     yolo=g["yolo"] and group("yolo", [YOLO_DIR / f"busi_m_s{s}" for s in SEEDS], ["busi:test", "uclm:all"]))
    rep["low_data"] = fr
    rows = []
    for f, d in fr.items():
        rows += [(f"{float(f) * 100:g}% WeVisOne gate", d["wevis"], "gate"), (f"{float(f) * 100:g}% YOLO cal", d["yolo"], "cal")]
    md.append("## 4. Low-data regime (fraction of the 368 BUSI training images)")
    md.append(table("", rows, ["busi:test", "uclm:all"]))

    # 5) multi-source
    ms = {}
    for tag in ("ms_bra", "ms_bra_uclm"):
        ms[tag] = dict(wevis=group("wevis", [WEVIS_DIR / f"{tag}_s{s}" for s in SEEDS], TESTS),
                       yolo=group("yolo", [YOLO_DIR / f"{tag}_s{s}" for s in SEEDS], TESTS))
    rep["multi_source"] = ms
    rows = [("BUSI only WeVisOne strong-aug + gate", g["wevis_aug"], "gate"), ("BUSI only YOLO cal", g["yolo"], "cal"),
            ("BUSI+BRA WeVisOne + gate", ms["ms_bra"]["wevis"], "gate"), ("BUSI+BRA YOLO cal", ms["ms_bra"]["yolo"], "cal"),
            ("BUSI+BRA+UCLMtrain WeVisOne + gate", ms["ms_bra_uclm"]["wevis"], "gate"),
            ("BUSI+BRA+UCLMtrain YOLO cal", ms["ms_bra_uclm"]["yolo"], "cal")]
    md.append("## 5. Multi-source training (BUS_UC never used; UCLM test patients never used)")
    md.append(table("", rows, ["busi:test", "uclm:test", "busuc:all", "busbra:test"]))

    # 5b) authoritative baselines: nnU-Net v2 (same split), MedSAM (box-prompted, zero-shot)
    nn = group("wevis", [RUNS / "nnunet" / "busi_s0"], TESTS)
    rep["nnunet"] = nn
    md.append("## 5b. Authoritative baselines on the same split and metric")
    md.append(table("nnU-Net v2 2d (nnUNetTrainer_250epochs, fold 0 = our train/val) vs WeVisOne vs YOLO", [
        ("WeVisOne raw", g["wevis"], "raw"), ("WeVisOne + gate", g["wevis"], "gate"),
        ("nnU-Net raw", nn, "raw"), ("nnU-Net + gate (val max-prob threshold)", nn, "gate"),
        ("YOLO val-calibrated", g["yolo"], "cal")], ["busi:test", "uclm:test", "uclm:all"]))
    md.append(table("", [("WeVisOne raw", g["wevis"], "raw"), ("nnU-Net raw", nn, "raw"), ("YOLO conf≥0.25", g["yolo"], "0.25")],
                    ["busuc:all", "busbra:test"]))
    ms_dir = RUNS / "medsam"
    dec = {}
    for spec in ["busi:test", "uclm:all", "uclm:test", "busuc:all", "busbra:test"]:
        row = {}
        med = wevis_records(ms_dir, spec, "gt_box")
        if med:
            row["MedSAM, GT boxes"] = summarize(med)["lesion_dice"]
        wv = [wevis_records(WEVIS_DIR / f"baseline_s{s}", spec, "gt_box") for s in SEEDS]
        wv = [summarize(r)["lesion_dice"] for r in wv if r]
        if wv:
            row["WeVisOne decoder, GT boxes"] = float(np.mean(wv))
        for s in SEEDS:
            rep_y = YOLO_DIR / f"busi_m_s{s}" / "eval" / "report.json"
            if not rep_y.exists():
                continue
            thr = json.loads(rep_y.read_text())["calibrated_threshold"]
            for label, recs in (("MedSAM, YOLO-cal boxes", wevis_records(ms_dir, spec, f"yolo{thr:g}_busi_m_s{s}")),
                                ("WeVisOne decoder, YOLO-cal boxes", hybrid_records(s, spec, "cal"))):
                if recs:
                    row.setdefault(label, []).append(summarize(recs)["lesion_dice"])
        dec[spec] = {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in row.items()}
    rep["decoder_comparison"] = dec
    if any(dec.values()):
        cols = ["MedSAM, GT boxes", "WeVisOne decoder, GT boxes", "MedSAM, YOLO-cal boxes", "WeVisOne decoder, YOLO-cal boxes"]
        md += ["### Box-prompted decoders (lesion Dice; MedSAM zero-shot, likely saw BUSI in pre-training)", "",
               "| Set | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
        for spec, row in dec.items():
            if row:
                md.append(f"| {spec} | " + " | ".join(f"{row[c]:.3f}" if c in row else "–" for c in cols) + " |")
        md.append("")

    # 5c) Yu et al. 2026 (QIMS) protocol replication
    paper = {}
    for s in range(3):
        run = YOLO_DIR / f"paper_busirand{s}_y11n"
        got = yolo_dets(run, f"busirand{s}:test") if (run / "eval").exists() else None
        if not got:
            continue
        recs, dets = got
        rows = dets_to_records(recs, dets, 0.25, strict=False)
        train_groups = {r["group"] for r in sources.rows(f"busirand{s}:train")}
        grp = {r["image"]: r["group"] for r in sources.rows(f"busirand{s}:test")}
        leak = [r for r in rows if grp[r["id"]] in train_groups]
        clean = [r for r in rows if grp[r["id"]] not in train_groups]
        out = dict(all=_dice_pair(rows, dets), leaked=_dice_pair(leak, dets), not_leaked=_dice_pair(clean, dets),
                   n_leaked=len(leak), n_not_leaked=len(clean))
        u = yolo_dets(run, "uclm:all")
        if u:
            urows = [r for r in dets_to_records(*u, 0.25, strict=False) if r["category"] != "normal"]
            out["uclm_lesions"] = _dice_pair(urows, u[1])
            out["uclm_normal_fpr"] = summarize(dets_to_records(*u, 0.25, strict=False))["fpr"]
        paper[s] = out
    rep["paper_repro_yu2026"] = paper
    if paper:
        md += ["## 5c. Replicating Yu et al. 2026 (QIMS) YOLO protocol",
               "Uncleaned BUSI lesions (647), image-level random 80/10/10, yolo11n-seg, 512 px, 50 epochs, conf ≥ 0.25. "
               "'leaked' = test image whose author-curated duplicate group also appears in train.", "",
               "| split seed | test per-image Dice | test pooled Dice | leaked (n) per-image | not leaked (n) per-image | "
               "UCLM lesions per-image | UCLM lesions pooled | UCLM normal FPR % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s, o in paper.items():
            u = o.get("uclm_lesions", (None, None))
            md.append(f"| {s} | {o['all'][0]:.3f} | {o['all'][1]:.3f} | {o['leaked'][0]:.3f} ({o['n_leaked']}) | "
                      f"{o['not_leaked'][0]:.3f} ({o['n_not_leaked']}) | "
                      + (f"{u[0]:.3f} | {u[1]:.3f} | {100 * o['uclm_normal_fpr']:.1f} |" if u[0] is not None else "– | – | – |"))
        md.append("")

    # 5d) synthetic domain-shift stress test (corruptions.py)
    import corruptions
    models = [("WeVisOne gate", "wevis", WEVIS_DIR / "baseline_s42", "gate"),
              ("WeVisOne strong-aug gate", "wevis", WEVIS_DIR / "aug_s42", "gate"),
              ("YOLO cal", "yolo", YOLO_DIR / "busi_m_s42", "cal"),
              ("nnU-Net gate", "wevis", RUNS / "nnunet" / "busi_s0", "gate")]
    corr = {}
    for fam in ["clean"] + corruptions.FAMILIES:
        for sev in ([0] if fam == "clean" else [1, 2]):
            suffix = "" if fam == "clean" else f"~{fam}{sev}"
            row = {}
            for label, kind, run, op in models:
                cells = {}
                for spec in ("busi:test", "uclm:test"):
                    recs = op_records(kind, run, spec + suffix, op) if run.exists() else None
                    if recs:
                        s = summarize(recs)
                        cells[spec] = dict(lesion_dice=s["lesion_dice"], fpr=s["fpr"], fp_k=s["fp_k"], normal_n=s["normal_n"])
                if cells:
                    row[label] = cells
            if row:
                corr[f"{fam}{sev if sev else ''}"] = row
    rep["corruptions"] = corr
    if len(corr) > 1:
        labels = [m[0] for m in models]
        md += ["## 5d. Synthetic domain-shift stress test (BUSI test 76 / UCLM held-out patients 331; seed 42 models)",
               "Families gain/speckle/blur overlap the strong-aug training augmentation (marked *). Thresholds fixed on BUSI val. "
               "Cells: BUSI lesion Dice / UCLM lesion Dice / UCLM normal FPR %.", "",
               "| Corruption | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for name, row in corr.items():
            fam = name.rstrip("012")
            star = "*" if fam in corruptions.IN_TRAIN else ""
            cells = []
            for lab in labels:
                c = row.get(lab, {})
                b, u = c.get("busi:test"), c.get("uclm:test")
                cells.append(" / ".join([f"{b['lesion_dice']:.3f}" if b else "–", f"{u['lesion_dice']:.3f}" if u else "–",
                                         f"{100 * u['fpr']:.1f}" if u and u["fpr"] is not None else "–"]))
            md.append(f"| {name}{star} | " + " | ".join(cells) + " |")
        md.append("")

    # 6) paired bootstrap, seed-averaged per image
    boot = {}
    pairs = [("WeVisOne gate vs YOLO cal", "wevis", "baseline_s{s}", "gate", "yolo", "busi_m_s{s}", "cal"),
             ("WeVisOne raw vs YOLO 0.25", "wevis", "baseline_s{s}", "raw", "yolo", "busi_m_s{s}", "0.25"),
             ("WeVisOne aug gate vs YOLO cal", "wevis", "aug_s{s}", "gate", "yolo", "busi_m_s{s}", "cal")]
    for spec in ["uclm:all", "uclm:test", "busuc:all"]:
        for label, ka, na, oa, kb, nb, ob in pairs:
            A = [op_records(ka, (WEVIS_DIR if ka == "wevis" else YOLO_DIR) / na.format(s=s), spec, oa) for s in SEEDS]
            B = [op_records(kb, (WEVIS_DIR if kb == "wevis" else YOLO_DIR) / nb.format(s=s), spec, ob) for s in SEEDS]
            A, B = [a for a in A if a], [b for b in B if b]
            if A and B:
                boot[f"{spec} | {label}"] = paired_bootstrap(A, B)
    if "wevis" in cv and "yolo" in cv:
        for op_a, op_b in [("gate", "cal"), ("raw", "0.25")]:
            boot[f"BUSI OOF | WeVisOne {op_a} vs YOLO {op_b}"] = paired_bootstrap([cv["wevis"][1][op_a]], [cv["yolo"][1][op_b]])
    rep["paired_bootstrap"] = boot
    if boot:
        md += ["## 6. Paired patient-cluster bootstrap (A − B; 5000 resamples)", "",
               "| Comparison | images / patients | ΔFPR (pp) [95% CI] | Δ lesion Dice [95% CI] |", "|---|---:|---:|---:|"]
        for k, b in boot.items():
            fpr = "–" if np.isnan(b["fpr_diff"]) else f"{100 * b['fpr_diff']:+.1f} [{100 * b['fpr_diff_ci'][0]:+.1f}, {100 * b['fpr_diff_ci'][1]:+.1f}]"
            md.append(f"| {k} | {b['n_images']} / {b['n_patients']} | {fpr} | "
                      f"{b['lesion_dice_diff']:+.3f} [{b['lesion_dice_diff_ci'][0]:+.3f}, {b['lesion_dice_diff_ci'][1]:+.3f}] |")
        md.append("")
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    (OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
