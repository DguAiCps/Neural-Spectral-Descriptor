#!/usr/bin/env python3
"""Graph-context identifiability, recomputed at the deployed 288D/416D operating point.

Backs the main-text claim that trajectory-neighborhood dissimilarity predicts
retrieval-key separation of invariant-key collisions (Prop. 2 / Assumption 1(iii)).

Protocol (per validation sequence, seed-1 pre-refinement 416D dump):
  d_hat = emb416[:, :288]  (unit-norm invariant key), f = emb416 (416D key)
  1. Collision cases: sampled far pairs (geo >= 25 m) with ||d_i - d_j|| <= 0.1.
  2. Neighborhood dissimilarity D_nb(i,j): symmetric Chamfer distance between the
     temporal-neighbor key multisets {d_hat_k : k in i+/-1..5} and likewise for j
     (mirrors the k=10 temporal window; multiset semantics of Assumption 1(iii)).
  3. Pearson r(D_nb, ||f_i - f_j||) over collision cases  -> identifiability corr.
  4. Permuted-context null: replace the context half of f with a random node's
     context (10 permutations) and recompute r  -> should collapse to ~0.
  5. Separation rate: share of collision pairs with ||f_i - f_j|| > 0.1, with the
     permuted-context rate as the dimension-mechanical baseline (this controls
     the 288D->416D dimension confound: concatenation can only add distance).
  6. Assumption 1(iii) empirical rate: share of collision pairs whose D_nb exceeds
     the q-quantile (q in {0.5, 0.9, 0.95}) of D_nb over *genuine revisit* pairs
     (same-place neighborhood variation = the noise floor for "same context").

Run: docker run --rm -v $REPO:/ws -w /ws nvcr.io/nvidia/pyg:26.01-py3 \
       python scripts/_verify_identifiability.py
Output: results/_alias_source/identifiability.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DUMP = REPO / "results/_final416_seed1"
OUT = REPO / "results/_alias_source"
SEQS = ["kitti_00", "kitti_05", "kitti_08", "nclt_2012-01-08", "nclt_2013-01-10",
        "helipr_Town01", "mulran_DCC03", "mulran_KAIST03", "mulran_Riverside03"]
EPS, GAMMA, POS_M, SKIP, W = 0.1, 25.0, 5.0, 30, 5
N_SAMPLE = 400_000
N_PERM = 10
RNG = np.random.default_rng(0)


def chamfer(A, B):
    """Symmetric Chamfer distance between two small point sets (rows)."""
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return 0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean())


def nb_sets(n):
    idx = []
    for i in range(n):
        w = [k for k in range(i - W, i + W + 1) if k != i and 0 <= k < n]
        idx.append(np.asarray(w, dtype=int))
    return idx


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for seq in SEQS:
        z = np.load(DUMP / f"{seq}.npz")
        f = z["emb416"].astype(np.float64)
        pos = z["poses"][:, :3, 3]
        n = len(f)
        d = f[:, :288]
        dn = np.linalg.norm(d, axis=1)
        d = d / np.clip(dn[:, None], 1e-12, None)
        ctx = f[:, 288:]
        nbs = nb_sets(n)

        # --- collision pairs (sampled far pairs within eps on the invariant key) ---
        ii = RNG.integers(0, n, N_SAMPLE)
        jj = RNG.integers(0, n, N_SAMPLE)
        far = np.linalg.norm(pos[ii] - pos[jj], axis=1) >= GAMMA
        ii, jj = ii[far], jj[far]
        dd = np.linalg.norm(d[ii] - d[jj], axis=1)
        col = dd <= EPS
        ci, cj = ii[col], jj[col]
        # dedup
        key = np.unique(np.minimum(ci, cj) * n + np.maximum(ci, cj))
        ci, cj = key // n, key % n
        if len(ci) < 20:
            out[seq] = {"n_collisions": int(len(ci)), "note": "too few collisions"}
            print(f"[{seq}] collisions={len(ci)} (skipped)", flush=True)
            continue
        if len(ci) > 20000:
            sel = RNG.choice(len(ci), 20000, replace=False)
            ci, cj = ci[sel], cj[sel]

        D_nb = np.asarray([chamfer(d[nbs[a]], d[nbs[b]]) for a, b in zip(ci, cj)])
        S = np.linalg.norm(f[ci] - f[cj], axis=1)
        r_real = float(np.corrcoef(D_nb, S)[0, 1])
        sep_real = float((S > EPS).mean() * 100)

        # --- permuted-context null ---
        r_perm, sep_perm = [], []
        for p in range(N_PERM):
            perm = RNG.permutation(n)
            fp = np.concatenate([d, ctx[perm]], axis=1)
            Sp = np.linalg.norm(fp[ci] - fp[cj], axis=1)
            r_perm.append(float(np.corrcoef(D_nb, Sp)[0, 1]))
            sep_perm.append(float((Sp > EPS).mean() * 100))

        # --- Assumption 1(iii) empirical rate ---
        # genuine revisit pairs: (j, earliest prior match) as in the eval protocol
        rev = []
        for j in range(SKIP, n):
            g = np.linalg.norm(pos[: j - SKIP + 1] - pos[j], axis=1)
            h = np.where(g < POS_M)[0]
            if len(h):
                rev.append((j, int(h[0])))
        D_rev = np.asarray([chamfer(d[nbs[a]], d[nbs[b]]) for a, b in rev]) if rev else np.asarray([np.nan])
        rates = {f"q{int(q*100)}": float((D_nb > np.quantile(D_rev, q)).mean() * 100)
                 for q in (0.5, 0.9, 0.95)}

        out[seq] = {
            "n": n, "n_collisions": int(len(ci)), "n_revisit_pairs": len(rev),
            "pearson_r": r_real,
            "pearson_r_perm_mean": float(np.mean(r_perm)),
            "pearson_r_perm_max_abs": float(np.max(np.abs(r_perm))),
            "sep_rate_pct": sep_real,
            "sep_rate_perm_mean_pct": float(np.mean(sep_perm)),
            "assumption1iii_rate_pct": rates,
            "D_nb_med_collision": float(np.median(D_nb)),
            "D_nb_med_revisit": float(np.median(D_rev)),
        }
        print(f"[{seq}] ncol={len(ci)} r={r_real:.3f} (perm {np.mean(r_perm):+.3f}) "
              f"sep={sep_real:.1f}% (perm {np.mean(sep_perm):.1f}%) "
              f"1(iii)@q90={rates['q90']:.1f}%", flush=True)

    json.dump(out, open(OUT / "identifiability.json", "w"), indent=1)
    print("wrote", OUT / "identifiability.json")


if __name__ == "__main__":
    main()
