#!/usr/bin/env python3
"""Non-learned sequence-context controls for the 288D invariant key.

Answers the causal question the learned-context ablation leaves open: is the
sparse-sensor gain of the 416D retrieval key due to the *learned* refiner, or
merely to *using sequence information at all*? Three closed-form controls, all
on the cached deployed 288D key, release retrieval protocol (candidates =
|i-j| >= 30, bidirectional):

  win_avg  : d_i replaced by the L2-normalized mean of {d_hat_j : j in i+/-5}
             (self included) -- plain temporal smoothing.
  concat   : [d_hat_i ; normalize(mean of d_hat over i+/-5 excl. self)] (576D)
             -- the structural analog of NSD's [key; context] concatenation
             with a non-learned "context" channel.
  seqmatch : SeqSLAM-style sequence-consistency score on the raw key cosine
             matrix: score(i,j) = max(forward, reverse) of the mean cosine over
             the aligned +/-5 window (reverse = query indices ascend while
             candidate indices descend) -- the strongest non-learned
             sequence-matching baseline, reverse-loop aware.

Reference points printed alongside: raw 288D key and (if the seed-1 dump
exists) the learned 416D retrieval key under the identical protocol.

Run: docker run --rm -v $REPO:/ws -w /ws nvcr.io/nvidia/pyg:26.01-py3 \
       python scripts/_verify_nonlearned_context.py
Output: results/_alias_source/nonlearned_context.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data/preprocessed_cross_sensor_operating"
DUMP416 = REPO / "results/_final416_seed1"
OUT = REPO / "results/_alias_source"
SEQS = {
    "kitti/00": "kitti_operating_00_layout60_stride1.npz",
    "kitti/05": "kitti_operating_05_layout60_stride1.npz",
    "kitti/08": "kitti_operating_08_layout60_stride1.npz",
    "nclt/2012-01-08": "nclt_operating_2012-01-08_layout60_stride1.npz",
    "nclt/2013-01-10": "nclt_operating_2013-01-10_layout60_stride1.npz",
    "helipr/Town01": "helipr_operating_Town01_layout60_stride1.npz",
    "mulran/DCC03": "mulran_operating_DCC03_layout60_stride1.npz",
    "mulran/KAIST03": "mulran_operating_KAIST03_layout60_stride1.npz",
    "mulran/Riverside03": "mulran_operating_Riverside03_layout60_stride1.npz",
}
QC = {"kitti/00": 632, "kitti/05": 377, "kitti/08": 235, "nclt/2012-01-08": 1834,
      "nclt/2013-01-10": 181, "helipr/Town01": 1586, "mulran/DCC03": 2344,
      "mulran/KAIST03": 2909, "mulran/Riverside03": 2796}
POS_M, SKIP, W = 5.0, 30, 5


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def find_queries(pos):
    q = []
    for j in range(SKIP, len(pos)):
        d = np.linalg.norm(pos[: j - SKIP + 1] - pos[j], axis=1)
        if (d < POS_M).any():
            q.append(j)
    return np.asarray(q, dtype=int)


def window_mean(d, incl_self):
    """Mean of unit keys over i+/-W (clipped at borders)."""
    n = len(d)
    acc = np.zeros_like(d)
    cnt = np.zeros((n, 1))
    offs = [k for k in range(-W, W + 1) if incl_self or k != 0]
    for k in offs:
        lo, hi = max(0, -k), min(n, n - k)
        acc[lo:hi] += d[lo + k:hi + k]
        cnt[lo:hi] += 1
    return l2(acc / cnt)


def r1_from_scores(S, pos, queries, larger_is_better=True):
    """Top-1 over release candidates |i-j|>=SKIP from a full score matrix."""
    n = len(pos)
    idx = np.arange(n)
    miss = 0
    for j in queries:
        cand = np.abs(idx - j) >= SKIP
        s = S[j, cand]
        t1 = int(np.argmax(s) if larger_is_better else np.argmin(s))
        geo = np.linalg.norm(pos[cand] - pos[j], axis=1)
        if geo[t1] >= POS_M:
            miss += 1
    return 1.0 - miss / max(len(queries), 1)


def seqmatch_scores(S):
    """max(forward, reverse) mean cosine over the aligned +/-W window."""
    n = S.shape[0]
    fwd = np.zeros_like(S)
    rev = np.zeros_like(S)
    cf = np.zeros_like(S)
    cr = np.zeros_like(S)
    for k in range(-W, W + 1):
        # forward: (i+k, j+k)
        si, sj = max(0, -k), max(0, -k)
        ei, ej = n - max(0, k), n - max(0, k)
        fwd[si:ei, sj:ej] += S[si + k:ei + k, sj + k:ej + k]
        cf[si:ei, sj:ej] += 1
        # reverse: (i+k, j-k)
        si, sj = max(0, -k), max(0, k)
        ei, ej = n - max(0, k), n - max(0, -k)
        rev[si:ei, sj:ej] += S[si + k:ei + k, sj - k:ej - k]
        cr[si:ei, sj:ej] += 1
    return np.maximum(fwd / np.clip(cf, 1, None), rev / np.clip(cr, 1, None))


def agg(per):
    ks = list(QC)
    rq = sum(per[k] * QC[k] for k in ks) / sum(QC.values())
    rs = float(np.mean([per[k] for k in ks]))
    mac = {}
    for k in ks:
        mac.setdefault(k.split("/")[0], []).append(per[k])
    macs = {s: float(np.mean(v)) for s, v in mac.items()}
    sig = float(np.sqrt(np.mean([(m - np.mean(list(macs.values()))) ** 2 for m in macs.values()])))
    return {"Rq": rq, "Rs": rs, "sigma": sig, "Rmin": float(min(macs.values()))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    variants = ["raw288", "win_avg", "concat", "seqmatch", "key416_seed1"]
    per = {v: {} for v in variants}
    for seq, fn in SEQS.items():
        z = np.load(CACHE / fn)
        d = l2(z["descriptors"].astype(np.float64))
        pos = z["poses"][:, :3, 3]
        queries = find_queries(pos)
        S = d @ d.T

        reps = {
            "raw288": None,  # scored via S directly
            "win_avg": window_mean(d, incl_self=True),
            "concat": l2(np.concatenate([d, window_mean(d, incl_self=False)], axis=1)),
        }
        per["raw288"][seq] = r1_from_scores(S, pos, queries)
        for v in ("win_avg", "concat"):
            per[v][seq] = r1_from_scores(reps[v] @ reps[v].T, pos, queries)
        per["seqmatch"][seq] = r1_from_scores(seqmatch_scores(S), pos, queries)

        d416 = DUMP416 / f"{seq.replace('/', '_')}.npz"
        if d416.exists():
            f = l2(np.load(d416)["emb416"].astype(np.float64))
            per["key416_seed1"][seq] = r1_from_scores(f @ f.T, pos, queries)

        print(f"[{seq}] " + " ".join(f"{v}={per[v].get(seq, float('nan')):.3f}" for v in variants),
              flush=True)

    out = {"per_seq": per, "aggregate": {v: agg(per[v]) for v in variants if len(per[v]) == 9}}
    json.dump(out, open(OUT / "nonlearned_context.json", "w"), indent=1)
    for v, a in out["aggregate"].items():
        print(f"{v:14} Rq={a['Rq']:.4f} Rs={a['Rs']:.4f} sig={a['sigma']:.4f} Rmin={a['Rmin']:.4f}")
    print("wrote", OUT / "nonlearned_context.json")


if __name__ == "__main__":
    main()
