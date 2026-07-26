#!/usr/bin/env python3
"""Paired uncertainty on the NSD-vs-SC++ query-weighted R@1 gap (TODO(student)).

Inputs
------
- SC++ per-query dumps:  results/per_query_baselines_regen/sc++/<DATASET>.json
  (fallback: results/per_query_baselines/sc++/). Schema: {"records":[{"query_idx",
  "success_at_k1", ...}]}, uppercase names (HELIPR_Town01, MULRAN_DCC03, ...).
- NSD B3 per-query dumps: results/per_query_b3/<seed>/{coarse,rerank}/<DATASET>.json
  (mixed-case names). Deployed selection: KITTI_* -> rerank (bev0.5_range0),
  all other sensors -> coarse (matches summarize_rank_k.py).

Tests
-----
1. McNemar (exact two-sided binomial on discordant pairs), pooled + per sequence.
2. Paired moving-block bootstrap on the per-query success difference, blocks of
   consecutive queries WITHIN each sequence (temporal correlation respected),
   B=10,000, percentile 95% CI on the query-weighted gap.

Usage:  python3 scripts/_paired_nsd_scpp_stats.py [--block 50] [--reps 10000]
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCPP_DIRS = [REPO / "results/per_query_baselines_regen/sc++",
             REPO / "results/per_query_baselines/sc++"]
NSD_ROOT = REPO / "results/per_query_b3"
SEEDS = ["seed1", "seed2", "seed3"]

# (canonical, sc++ alias, nsd alias, deployed NSD channel)
SEQS = [
    ("KITTI_00", "KITTI_00", "KITTI_00", "rerank"),
    ("KITTI_05", "KITTI_05", "KITTI_05", "rerank"),
    ("KITTI_08", "KITTI_08", "KITTI_08", "rerank"),
    ("NCLT_2012-01-08", "NCLT_2012-01-08", "NCLT_2012-01-08", "coarse"),
    ("NCLT_2013-01-10", "NCLT_2013-01-10", "NCLT_2013-01-10", "coarse"),
    ("HeLiPR_Town01", "HELIPR_Town01", "HeLiPR_Town01", "coarse"),
    ("MulRan_DCC03", "MULRAN_DCC03", "MulRan_DCC03", "coarse"),
    ("MulRan_KAIST03", "MULRAN_KAIST03", "MulRan_KAIST03", "coarse"),
    ("MulRan_Riverside03", "MULRAN_Riverside03", "MulRan_Riverside03", "coarse"),
]


def load_records(path: Path) -> dict:
    data = json.load(open(path))
    return {int(r["query_idx"]): bool(r["success_at_k1"]) for r in data["records"]}


def load_scpp(alias: str) -> dict:
    for d in SCPP_DIRS:
        p = d / f"{alias}.json"
        if p.exists():
            return load_records(p)
    raise FileNotFoundError(f"SC++ dump missing for {alias}")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on discordant counts (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    logp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            - n * math.log(2.0) for i in range(n + 1)]
    p = sum(math.exp(lp) for i, lp in enumerate(logp) if i <= k or i >= n - k)
    return min(1.0, p)


def block_bootstrap(pairs_by_seq, block: int, reps: int, rng) -> np.ndarray:
    """Moving-block bootstrap of the query-weighted gap. pairs_by_seq:
    list of (nsd 0/1 array, scpp 0/1 array) in temporal (query_idx) order."""
    total_n = sum(len(a) for a, _ in pairs_by_seq)
    gaps = np.empty(reps)
    starts_by_seq = [max(1, len(a) - block + 1) for a, _ in pairs_by_seq]
    for r in range(reps):
        hits_nsd = hits_scpp = 0
        for (a, b), n_starts in zip(pairs_by_seq, starts_by_seq):
            n = len(a)
            need = math.ceil(n / block)
            starts = rng.integers(0, n_starts, size=need)
            idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
            hits_nsd += a[idx].sum()
            hits_scpp += b[idx].sum()
        gaps[r] = (hits_nsd - hits_scpp) / total_n
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=50)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--output", type=Path,
                    default=REPO / "results/paired_nsd_scpp_stats.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    out = {"block": args.block, "reps": args.reps, "seeds": {}}
    for seed in SEEDS:
        pairs_by_seq, per_seq = [], {}
        B = C = hits_nsd = hits_scpp = total = 0
        for canon, al_s, al_n, channel in SEQS:
            scpp = load_scpp(al_s)
            nsd = load_records(NSD_ROOT / seed / channel / f"{al_n}.json")
            qidx = sorted(scpp.keys())
            assert set(qidx) == set(nsd.keys()), f"query sets differ on {canon}"
            a = np.array([nsd[i] for i in qidx], dtype=np.int64)
            s = np.array([scpp[i] for i in qidx], dtype=np.int64)
            pairs_by_seq.append((a, s))
            b = int(((a == 1) & (s == 0)).sum())
            c = int(((a == 0) & (s == 1)).sum())
            B += b; C += c
            hits_nsd += int(a.sum()); hits_scpp += int(s.sum()); total += len(a)
            per_seq[canon] = {"n": len(a), "nsd_r1": float(a.mean()),
                              "scpp_r1": float(s.mean()), "b_nsd_only": b,
                              "c_scpp_only": c, "mcnemar_p": mcnemar_exact(b, c)}
        gap = (hits_nsd - hits_scpp) / total
        gaps = block_bootstrap(pairs_by_seq, args.block, args.reps, rng)
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        p_boot = 2.0 * min((gaps <= 0).mean(), (gaps >= 0).mean())
        out["seeds"][seed] = {
            "nsd_rq": hits_nsd / total, "scpp_rq": hits_scpp / total,
            "gap_pp": 100 * gap, "boot_ci95_pp": [100 * lo, 100 * hi],
            "boot_p": float(min(1.0, max(p_boot, 2.0 / args.reps))),
            "mcnemar_pooled": {"b_nsd_only": B, "c_scpp_only": C,
                               "p": mcnemar_exact(B, C)},
            "per_sequence": per_seq,
        }
        print(f"[{seed}] NSD {hits_nsd/total:.4f} vs SC++ {hits_scpp/total:.4f}  "
              f"gap {100*gap:+.2f}pp  CI95 [{100*lo:+.2f},{100*hi:+.2f}]  "
              f"boot p<={out['seeds'][seed]['boot_p']:.4g}  "
              f"McNemar b={B} c={C} p={mcnemar_exact(B, C):.3g}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=1)
    print("saved", args.output)


if __name__ == "__main__":
    main()
