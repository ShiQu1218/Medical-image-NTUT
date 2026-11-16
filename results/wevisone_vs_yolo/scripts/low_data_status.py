"""Audit all nine low-data WeVisOne runs against expected data IDs and training budget."""
from __future__ import annotations

import csv
import json

import common
import sources

FRACTIONS = ("0.1", "0.25", "0.5")
SEEDS = (42, 43, 44)
SPECS = ("busi:val", "busi:test", "uclm:all")


def audit():
    expected = {spec: {r["image"] if spec.startswith("busi:") else r["Image"]
                       for r in sources.rows(spec)} for spec in SPECS}
    out = {}
    for f in FRACTIONS:
        for seed in SEEDS:
            run = common.RUNS / "wevis" / f"frac{f}_s{seed}"
            issues, sets = [], {}
            args = json.loads((run / "args.json").read_text()) if (run / "args.json").exists() else {}
            for k, v in dict(seed=seed, train_spec=f"busi:train@{f}/{seed}", val_spec="busi:val",
                             total_steps=230, n_vals=5).items():
                if args.get(k) != v:
                    issues.append(f"{k}: expected {v}, got {args.get(k)}")
            history = list(csv.DictReader((run / "history.csv").open(newline=""))) if (run / "history.csv").exists() else []
            vals = [r for r in history if r.get("val_dice")]
            steps = max((int(r["step"]) for r in history), default=0)
            if steps != 230 or len(vals) != 5:
                issues.append(f"training: {steps}/230 steps, {len(vals)}/5 validations")
            for name in ("TRAIN_DONE", "best_decoder.pt", "best_lora/adapter_model.safetensors"):
                if not (run / name).exists():
                    issues.append(f"missing {name}")
            for spec in SPECS:
                path = run / "eval" / f"{spec.replace(':', '_')}_generate.jsonl"
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
                ids = {r["id"] for r in rows}
                valid = ids == expected[spec] and len(rows) == len(ids)
                sets[spec] = dict(n=len(ids), expected_n=len(expected[spec]), complete=valid)
                if not valid:
                    issues.append(f"incomplete {spec}: {len(ids)}/{len(expected[spec])} IDs")
            out[run.name] = dict(complete=not issues, issues=issues, steps=steps, validations=len(vals), sets=sets)
    return out


def main():
    result = audit()
    out = common.RUNS / "comparison" / "low_data_status.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, r in result.items():
        print(f"{name}: {'complete' if r['complete'] else 'pending'}; steps={r['steps']}, validations={r['validations']}")
    return 0 if all(r["complete"] for r in result.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
