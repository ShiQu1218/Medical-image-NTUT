"""Near-duplicate screen across datasets: cosine similarity of z-normalised 32x32 thumbnails."""
import json
import sys

import cv2
import numpy as np

import common
import sources


def thumb(item):
    g = np.asarray(item["image"].convert("L"), np.float32)
    t = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).flatten()
    t = t - t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def feats(spec):
    ids, vs = [], []
    for it in sources.items(spec):
        ids.append(it["id"])
        vs.append(thumb(it))
    return ids, np.stack(vs)


def main(query_spec, ref_specs, thr=0.95):
    qi, qv = feats(query_spec)
    hits = []
    for ref in ref_specs:
        ri, rv = feats(ref)
        sim = qv @ rv.T
        j = sim.argmax(1)
        for a in np.where(sim.max(1) > thr)[0]:
            hits.append(dict(query=qi[a], ref=ri[j[a]], sim=float(sim[a, j[a]])))
        print(query_spec, "vs", ref, "max sims p50/p99/max:",
              *np.round(np.percentile(sim.max(1), [50, 99, 100]), 3), "hits>", thr, sum(h["ref"] in ri for h in hits))
    return hits


if __name__ == "__main__":
    out = {}
    out["busuc"] = main("busuc:all", ["busi:train+busi:val+busi:test", "uclm:all", "busbra:all"])
    out["busbra"] = main("busbra:all", ["busi:train+busi:val+busi:test", "uclm:all"])
    (common.RUNS / "dup_check.json").write_text(json.dumps(out, indent=2))
