#!/usr/bin/env python3
"""Aliasing-source ladder: separate phase-removal from magnitude-compression.

The paper attributes spectral aliasing to phase removal, but the deployed 288D
key adds a second lossy stage (octave binning + mean/std pooling, 16x181 -> 288).
This script decomposes the non-injectivity along the encoder ladder, on the
cached full FFT magnitudes of the 9 validation sequences:

  full2896 : flattened 16x181 log-magnitudes, L2-normalized   (phase removed, no binning)
  bin288   : deployed octave mean+std key (cached descriptors) (phase removed + binned)
  mean144  : octave mean-only key                              (harsher pooling)

Per representation and sequence:
  - R@1 (standard revisit protocol: <5 m, >=30-frame gap)
  - AliasRate(eps=0.1, gamma=25) on 200k sampled far pairs (legacy fixed-eps)
  - AliasRate at a *recall-matched* eps: eps_D = the quantile of same-place
    best-match distances that makes P(same-place accepted) equal to the deployed
    (bin288, eps=0.1) acceptance -> dimension/scale-free comparison
  - per-query miss decomposition (collision / spread / near-field)
  - pair tracking: of the far pairs colliding under bin288, the share still
    colliding under full2896 at its matched eps (= phase/projection-inherent)
    vs. separated by the full spectrum (= created by binning compression)

Also: eps-sensitivity sweep of the bin288 miss decomposition (B4), plus the
416D retrieval key (seed-1 dump) at the default eps for reference.

Run inside the pyg container (numpy-only, CPU):
  docker run --rm -v $REPO:/ws -w /ws nvcr.io/nvidia/pyg:26.01-py3 \
    python scripts/_verify_alias_source_ladder.py
Output: results/_alias_source/ladder.json
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
EPS, GAMMA = 0.1, 25.0
POS_M, SKIP = 5.0, 30
N_FAR = 200_000
EPS_SWEEP = [0.05, 0.075, 0.1, 0.15, 0.2]
OCTAVE = [0, 1, 2, 4, 8, 16, 32, 64, 128, 181]
RNG = np.random.default_rng(0)


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def octave_stats(mags, with_std=True):
    """mags: (n,16,181) -> per-channel-L2-normalized [mu(144) | sigma(144)]."""
    n = mags.shape[0]
    mus, sds = [], []
    for b in range(len(OCTAVE) - 1):
        seg = mags[:, :, OCTAVE[b]:OCTAVE[b + 1]]
        mu = seg.mean(axis=2)                      # (n,16)
        mus.append(mu)
        if with_std:
            sds.append(np.sqrt(((seg - mu[:, :, None]) ** 2).mean(axis=2) + 1e-8))
    mu = np.stack(mus, axis=2).reshape(n, -1)      # (n,144) elevation-major
    if not with_std:
        return l2(mu)
    sd = np.stack(sds, axis=2).reshape(n, -1)
    return np.concatenate([l2(mu), l2(sd)], axis=1)


def find_queries(pos):
    q = []
    for j in range(SKIP, len(pos)):
        d = np.linalg.norm(pos[: j - SKIP + 1] - pos[j], axis=1)
        if (d < POS_M).any():
            q.append(j)
    return np.asarray(q, dtype=int)


def eval_rep(emb, pos, queries, far_i, far_j, eps_default=EPS):
    """emb must be globally L2-normalized."""
    res = {}
    n = len(emb)
    # per-query: release protocol candidates = ALL frames with |i-j| >= SKIP
    # (run_kitti_operating_point._topk_cosine); emb is unit-norm, so
    # ||a-b|| = sqrt(2-2*cos) and cosines come from one BLAS matmul.
    sims = emb[queries] @ emb.T  # (n_q, n)
    idx = np.arange(n)
    best_same, top1_geo, miss = [], [], []
    for qi, j in enumerate(queries):
        cand = np.abs(idx - j) >= SKIP
        d = np.sqrt(np.clip(2.0 - 2.0 * sims[qi, cand], 0.0, None))
        geo = np.linalg.norm(pos[cand] - pos[j], axis=1)
        same = geo < POS_M
        best_same.append(d[same].min())
        t1 = int(np.argmin(d))
        top1_geo.append(geo[t1])
        miss.append(not same[t1])
    best_same = np.asarray(best_same)
    top1_geo = np.asarray(top1_geo)
    miss = np.asarray(miss)
    res["r1"] = float(1.0 - miss.mean())
    res["same_med"] = float(np.median(best_same))
    # far-pair distances
    dfar = np.linalg.norm(emb[far_i] - emb[far_j], axis=1)
    res["aliasrate_fixed"] = float((dfar <= eps_default).mean())
    res["_best_same"] = best_same
    res["_top1_geo"] = top1_geo
    res["_miss"] = miss
    res["_dfar"] = dfar
    return res


def decompose(res, eps):
    """Miss decomposition at threshold eps -> shares of misses."""
    m = res["_miss"]
    if m.sum() == 0:
        return {"miss_pct": 0.0, "collision": 0.0, "spread": 0.0, "near": 0.0}
    spread = res["_best_same"][m] > eps
    coll = (~spread) & (res["_top1_geo"][m] >= GAMMA)
    near = (~spread) & (res["_top1_geo"][m] < GAMMA)
    return {
        "miss_pct": float(m.mean() * 100),
        "collision": float(coll.mean() * 100),
        "spread": float(spread.mean() * 100),
        "near": float(near.mean() * 100),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for seq, fn in SEQS.items():
        z = np.load(CACHE / fn)
        mags = z["fft_magnitudes"].astype(np.float64)
        pos = z["poses"][:, :3, 3]
        _desc = z["descriptors"].astype(np.float64)
        if _desc.shape[1] != 288:
            mus0, sds0 = [], []
            for _b in range(len(OCTAVE) - 1):
                _seg = mags[:, :, OCTAVE[_b]:OCTAVE[_b + 1]]
                _mu = _seg.mean(axis=2)
                mus0.append(_mu)
                sds0.append(np.sqrt(((_seg - _mu[:, :, None]) ** 2).mean(axis=2) + 1e-8))
            _desc = np.concatenate([np.stack(mus0, 2).reshape(len(mags), -1),
                                    np.stack(sds0, 2).reshape(len(mags), -1)], axis=1)
        reps = {
            "bin288": l2(_desc),
            "full2896": l2(mags.reshape(len(mags), -1)),
            "mean144": octave_stats(mags, with_std=False),
        }
        # recon sanity: raw (un-normalized) mu/sigma must match the cached key
        mus, sds = [], []
        for b in range(len(OCTAVE) - 1):
            seg = mags[:, :, OCTAVE[b]:OCTAVE[b + 1]]
            mu = seg.mean(axis=2)
            mus.append(mu)
            sds.append(np.sqrt(((seg - mu[:, :, None]) ** 2).mean(axis=2) + 1e-8))
        rec = np.concatenate([np.stack(mus, 2).reshape(len(mags), -1),
                              np.stack(sds, 2).reshape(len(mags), -1)], axis=1)
        if z["descriptors"].shape[1] == 288:
            recon_cos = float(np.abs(rec - z["descriptors"]).max())  # max abs diff, ~1e-4
        else:
            recon_cos = -1.0  # cache stores a non-288D descriptor; bin288 was reconstructed

        d416 = DUMP416 / f"{seq.replace('/', '_')}.npz"
        if d416.exists():
            reps["key416_seed1"] = l2(np.load(d416)["emb416"].astype(np.float64))

        queries = find_queries(pos)
        # shared far-pair sample
        nn = len(pos)
        ii = RNG.integers(0, nn, N_FAR * 2)
        jj = RNG.integers(0, nn, N_FAR * 2)
        keep = np.linalg.norm(pos[ii] - pos[jj], axis=1) >= GAMMA
        far_i, far_j = ii[keep][:N_FAR], jj[keep][:N_FAR]

        r = {k: eval_rep(e, pos, queries, far_i, far_j) for k, e in reps.items()}

        # recall-matched eps: acceptance of deployed key at eps=0.1
        q_acc = float((r["bin288"]["_best_same"] <= EPS).mean())
        seq_out = {"n": int(nn), "n_q": int(len(queries)), "recon_cos": recon_cos,
                   "matched_acceptance": q_acc, "reps": {}}
        for k in reps:
            eps_m = float(np.quantile(r[k]["_best_same"], q_acc)) if 0 < q_acc < 1 \
                else (EPS if k == "bin288" else float(np.quantile(r[k]["_best_same"], 0.9)))
            rr = {kk: vv for kk, vv in r[k].items() if not kk.startswith("_")}
            rr["eps_matched"] = eps_m
            rr["aliasrate_matched"] = float((r[k]["_dfar"] <= eps_m).mean())
            rr["decomp_default"] = decompose(r[k], EPS)
            rr["decomp_matched"] = decompose(r[k], eps_m)
            seq_out["reps"][k] = rr

        # pair tracking: bin288 collisions under the full spectrum
        c288 = r["bin288"]["_dfar"] <= EPS
        if c288.sum() > 0:
            eps_full = seq_out["reps"]["full2896"]["eps_matched"]
            still = r["full2896"]["_dfar"][c288] <= eps_full
            seq_out["pair_tracking"] = {
                "n_collisions_288": int(c288.sum()),
                "still_colliding_full_pct": float(still.mean() * 100),
                "separated_by_full_pct": float((~still).mean() * 100),
            }
        # B4: eps sweep of the bin288 (and 416D) miss decomposition
        seq_out["eps_sweep"] = {
            k: {str(e): decompose(r[k], e) for e in EPS_SWEEP}
            for k in reps if k in ("bin288", "key416_seed1")
        }
        out[seq] = seq_out
        print(f"[{seq}] n={nn} q={len(queries)} recon={recon_cos:.6f} "
              f"| R@1 288={seq_out['reps']['bin288']['r1']:.3f} "
              f"full={seq_out['reps']['full2896']['r1']:.3f} "
              f"mean144={seq_out['reps']['mean144']['r1']:.3f} "
              f"| AR-fixed 288={seq_out['reps']['bin288']['aliasrate_fixed']*100:.2f}% "
              f"| AR-matched full={seq_out['reps']['full2896']['aliasrate_matched']*100:.3f}% "
              f"288={seq_out['reps']['bin288']['aliasrate_matched']*100:.3f}%", flush=True)

    json.dump(out, open(OUT / "ladder.json", "w"), indent=1)
    print("wrote", OUT / "ladder.json")


if __name__ == "__main__":
    main()
