#!/usr/bin/env python3
"""Fusion-tuple selection for the pd protocol, mirroring the paper-main
selection mechanism in scripts/_final_fix8_agg.py (query-weighted pooled R@1,
summed across the 3 seeds, argmax over the shared WEIGHTS grid) -- but the
selection pool is the pd NEW VAL set (the only split tuple selection may see;
test remains untouched):

  pd    K09 N12PD RA3 SJ3      pdnn  K09 RA3 SJ3

Reads results/pd2026/branch_fusion_<variant>_seed{1,2,3}.json and prints, per
variant x operating point (pbev128_main = NSD, hspec12 = NSD-H): the selected
tuple, per-sequence 3-seed mean R@1, Rq (query-weighted, mean +- sd across
seeds), Rs (sequence mean), sensor sigma, Rmin. Writes the frozen selection to
results/pd2026/tuple_selection.json.
"""
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results" / "pd2026"

SETS = {
    "pd": ["K09", "N12PD", "RA3", "SJ3"],
    "pdnn": ["K09", "RA3", "SJ3"],
}
SENSOR = {"K09": "kitti", "N12PD": "nclt", "RA3": "helipr", "SJ3": "mulran"}
BRANCHES = ["pbev128_main", "hspec12"]


def agg(rep, v, tags, w):
    n = d = 0.0
    for t in tags:
        x = rep[t]["variants"][v]
        n += x["n_queries"] * x[w]
        d += x["n_queries"]
    return n / d


out = {}
for variant, tags in SETS.items():
    files = [R / f"branch_fusion_{variant}_seed{s}.json" for s in (1, 2, 3)]
    try:
        reps = [json.loads(f.read_text()) for f in files]
    except FileNotFoundError as e:
        print(f"--- {variant}: MISSING {e}")
        continue
    for branch in BRANCHES:
        ok = [r for r in reps
              if all(t in r and branch in r[t].get("variants", {}) for t in tags)]
        if len(ok) < len(reps):
            print(f"--- {variant}/{branch}: only {len(ok)}/3 seeds complete")
        if not ok:
            continue
        ws = [k for k in ok[0][tags[0]]["variants"][branch] if k.startswith("(")]
        w = max(ws, key=lambda x: sum(agg(r, branch, tags, x) for r in ok))
        per = {t: statistics.mean(r[t]["variants"][branch][w] for r in ok)
               for t in tags}
        q = {t: ok[0][t]["variants"][branch]["n_queries"] for t in tags}
        rq = [agg(r, branch, tags, w) for r in ok]
        rs = statistics.mean(per.values())
        sm = {}
        for t in tags:
            sm.setdefault(SENSOR[t], []).append(per[t])
        means = [statistics.mean(v) for v in sm.values()]
        mu = statistics.mean(means)
        sigma = (sum((m - mu) ** 2 for m in means) / len(means)) ** 0.5
        print(f"--- {variant}/{branch} ({len(ok)} seeds) w={w}")
        print(f"  Rq={statistics.mean(rq):.5f} +- "
              f"{statistics.stdev(rq) if len(rq) > 1 else 0:.5f}  "
              f"Rs={rs:.4f}  sigma={sigma:.4f}  Rmin={min(means):.4f}")
        print("  " + "  ".join(f"{t}={per[t]:.4f}" for t in tags))
        print("  q: " + " ".join(str(q[t]) for t in tags))
        out[f"{variant}/{branch}"] = {
            "selected_tuple": w, "n_seeds": len(ok),
            "selection_pool": tags,
            "Rq_mean": statistics.mean(rq),
            "Rq_sd": statistics.stdev(rq) if len(rq) > 1 else 0.0,
            "Rq_per_seed": rq,
            "Rs": rs, "sensor_sigma": sigma, "Rmin": min(means),
            "per_sequence_mean": per, "n_queries": q,
        }
(R / "tuple_selection.json").write_text(json.dumps(out, indent=2))
print("PD_TUPLE_SELECT_DONE")
