#!/usr/bin/env python3
"""Aggregate the A/B2 experiment: per-sequence best of {coarse, rerank grid}.

Mirrors the release summarizer's per-sequence key selection (raw vs fusion),
so numbers are comparable to the paper's Table 1 protocol. Reads
results/_remetrize_twopass/{seedN,rerank_seedN}.json and prints per-sequence
3-seed means plus aggregate rows for the P / A / B2 variants.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "results/_remetrize_twopass"
SEEDS = ["seed1", "seed2", "seed3"]
VARIANTS = ["P", "A", "B2"]
ORDER = ["kitti/00", "kitti/05", "kitti/08", "nclt/2012-01-08", "nclt/2013-01-10",
         "helipr/Town01", "mulran/DCC03", "mulran/KAIST03", "mulran/Riverside03"]


def aggregates(per_seq, counts):
    vals = np.array([per_seq[s] for s in ORDER])
    n = np.array([counts[s] for s in ORDER], dtype=float)
    macro = {}
    for s in ORDER:
        macro.setdefault(s.split("/")[0], []).append(per_seq[s])
    macros = np.array([np.mean(v) for v in macro.values()])
    return {"R^q": float((vals * n).sum() / n.sum()), "R^s": float(vals.mean()),
            "sigma_cross": float(np.sqrt(((macros - macros.mean()) ** 2).mean())),
            "R_min": float(macros.min())}


def main():
    import sys
    suffix = "_raw" if "raw" in sys.argv[1:] else ""
    do_global = "global" in sys.argv[1:]

    best = {v: {s: [] for s in ORDER} for v in VARIANTS}
    grids = {v: {s: [] for s in ORDER} for v in VARIANTS}  # dict-per-seed of weight->R@1
    counts = None
    for seed in SEEDS:
        coarse = json.load(open(DIR / f"{seed}.json"))
        rerank = json.load(open(DIR / f"rerank{suffix}_{seed}.json"))
        counts = {s: rerank[s]["n_q"] for s in ORDER}
        for v in VARIANTS:
            for s in ORDER:
                fused = max(rerank[s][v].values())
                best[v][s].append(max(coarse[v]["per_seq"][s], fused))
                grids[v][s].append(rerank[s][v])

    if do_global:
        # one (w_bev, w_range) pair fixed across ALL sequences/sensors,
        # chosen to maximize the baseline P's 3-seed query-weighted mean.
        keys = sorted(grids["P"][ORDER[0]][0].keys())
        n = np.array([counts[s] for s in ORDER], dtype=float)

        def rq(v, key):
            vals = np.array([[grids[v][s][i][key] for s in ORDER] for i in range(len(SEEDS))])
            return float((vals.mean(axis=0) * n).sum() / n.sum())

        best_key = max(keys, key=lambda k: rq("P", k))
        print(f"[global fixed weight] selected by P: {best_key}")
        for v in VARIANTS:
            per_seq = {s: float(np.mean([grids[v][s][i][best_key] for i in range(len(SEEDS))]))
                       for s in ORDER}
            agg = aggregates(per_seq, counts)
            print(f"  {v}: R^q={agg['R^q']:.4f} R^s={agg['R^s']:.4f} "
                  f"sigma={agg['sigma_cross']:.4f} R_min={agg['R_min']:.4f}")
        print()

    print(f"{'sequence':<20}" + "".join(f"{v:>10}" for v in VARIANTS) + "   (3-seed mean, per-seq best)")
    for s in ORDER:
        print(f"{s:<20}" + "".join(f"{np.mean(best[v][s]):10.3f}" for v in VARIANTS))
    print()
    for v in VARIANTS:
        per_seed_aggs = []
        for i, _ in enumerate(SEEDS):
            per_seq = {s: best[v][s][i] for s in ORDER}
            per_seed_aggs.append(aggregates(per_seq, counts))
        mean_seq = {s: float(np.mean(best[v][s])) for s in ORDER}
        agg = aggregates(mean_seq, counts)
        rng = {k: (min(a[k] for a in per_seed_aggs), max(a[k] for a in per_seed_aggs))
               for k in ("R^q", "R_min", "sigma_cross")}
        print(f"{v}: R^q={agg['R^q']:.4f} [{rng['R^q'][0]:.3f},{rng['R^q'][1]:.3f}]  "
              f"R^s={agg['R^s']:.4f}  sigma={agg['sigma_cross']:.4f} "
              f"[{rng['sigma_cross'][0]:.3f},{rng['sigma_cross'][1]:.3f}]  "
              f"R_min={agg['R_min']:.4f} [{rng['R_min'][0]:.3f},{rng['R_min'][1]:.3f}]")


if __name__ == "__main__":
    main()
