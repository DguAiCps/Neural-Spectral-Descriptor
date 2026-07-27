#!/usr/bin/env python3
"""AliasRate vs. vector size for the magnitude-only key (rate_ladder companion).

Sweeps the same size ladder as the paper's rate_ladder table on the cached
16x181 log-magnitude spectra of the 9 validation sequences:

  full2896  : flattened 16x181, L2-normalized
  trunc1440 : first 90 frequency columns   (16x90)
  trunc720  : first 45                     (16x45)
  trunc352  : first 22                     (16x22)
  trunc176  : first 11                     (16x11)
  trunc80   : first 5                      (16x5)
  bin288    : deployed octave mean+std key (cached descriptors)

Per representation and sequence:
  - R@1 (revisit protocol <5 m, >=30-frame gap)  -> anchor vs. rate_ladder
  - AliasRate at fixed eps=0.1                   -> anchor vs. ladder.json
  - AliasRate at recall-matched eps_D (same-place acceptance matched to the
    deployed bin288 key at eps=0.1; protocol of _verify_alias_source_ladder.py)

Far pairs are drawn with the same RNG stream/order as
_verify_alias_source_ladder.py, so fixed-eps bin288 values must match
results/_alias_source/ladder.json exactly.

Host-runnable (numpy-only, CPU):
  python3 scripts/_aliasrate_size_sweep.py
Output: results/_alias_source/aliasrate_size_sweep.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data/preprocessed_cross_sensor_operating"
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
TRUNC = {"trunc1440": 90, "trunc720": 45, "trunc352": 22, "trunc176": 11, "trunc80": 5}
RNG = np.random.default_rng(0)


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def find_queries(pos):
    n = len(pos)
    idx = np.arange(n)
    queries = []
    for j in range(n):
        cand = np.abs(idx - j) >= SKIP
        earlier = cand & (idx < j)
        if earlier.any() and (np.linalg.norm(pos[earlier] - pos[j], axis=1) < POS_M).any():
            queries.append(j)
    return np.asarray(queries)


def eval_rep(emb, geo_masks, far_i, far_j):
    """geo_masks: per-query (cand, same, geo) precomputed once per sequence."""
    best_same, top1_geo, miss = [], [], []
    for (j, cand, same, geo) in geo_masks:
        sims = emb[cand] @ emb[j]
        d = np.sqrt(np.clip(2.0 - 2.0 * sims, 0.0, None))
        best_same.append(d[same].min())
        t1 = int(np.argmin(d))
        top1_geo.append(geo[t1])
        miss.append(not same[t1])
    dfar = np.linalg.norm(emb[far_i] - emb[far_j], axis=1)
    return {
        "_best_same": np.asarray(best_same),
        "_miss": np.asarray(miss),
        "_dfar": dfar,
        "r1": float(1.0 - np.mean(miss)),
        "aliasrate_fixed": float((dfar <= EPS).mean()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for seq, fn in SEQS.items():
        z = np.load(CACHE / fn)
        mags = z["fft_magnitudes"].astype(np.float64)
        pos = z["poses"][:, :3, 3]
        nn = len(pos)

        reps = {"full2896": l2(mags.reshape(nn, -1))}
        for name, k in TRUNC.items():
            reps[name] = l2(mags[:, :, :k].reshape(nn, -1))
        reps["bin288"] = l2(z["descriptors"].astype(np.float64))

        queries = find_queries(pos)
        idx = np.arange(nn)
        geo_masks = []
        for j in queries:
            cand = np.abs(idx - j) >= SKIP
            geo = np.linalg.norm(pos[cand] - pos[j], axis=1)
            geo_masks.append((j, cand, geo < POS_M, geo))

        # identical far-pair stream to _verify_alias_source_ladder.py
        ii = RNG.integers(0, nn, N_FAR * 2)
        jj = RNG.integers(0, nn, N_FAR * 2)
        keep = np.linalg.norm(pos[ii] - pos[jj], axis=1) >= GAMMA
        far_i, far_j = ii[keep][:N_FAR], jj[keep][:N_FAR]

        r = {k: eval_rep(e, geo_masks, far_i, far_j) for k, e in reps.items()}

        q_acc = float((r["bin288"]["_best_same"] <= EPS).mean())
        seq_out = {"n": int(nn), "n_q": int(len(queries)),
                   "matched_acceptance": q_acc, "reps": {}}
        for k in reps:
            eps_m = float(np.quantile(r[k]["_best_same"], q_acc)) if 0 < q_acc < 1 \
                else (EPS if k == "bin288" else float(np.quantile(r[k]["_best_same"], 0.9)))
            seq_out["reps"][k] = {
                "r1": r[k]["r1"],
                "aliasrate_fixed": r[k]["aliasrate_fixed"],
                "eps_matched": eps_m,
                "aliasrate_matched": float((r[k]["_dfar"] <= eps_m).mean()),
            }
        out[seq] = seq_out
        line = " ".join(f"{k}:AR={seq_out['reps'][k]['aliasrate_matched']*100:.3f}%"
                        for k in reps)
        print(f"[{seq}] n={nn} q={len(queries)} acc={q_acc:.3f} | {line}", flush=True)

    # summaries: query-weighted R@1 and pair-pooled AliasRate, 8-seq (rate_ladder
    # protocol, Town01 excluded) and all-9
    for tag, seqs in (("8seq_no_town01", [s for s in SEQS if s != "helipr/Town01"]),
                      ("9seq", list(SEQS))):
        summ = {}
        for k in list(TRUNC) + ["full2896", "bin288"]:
            nq = sum(out[s]["n_q"] for s in seqs)
            r1 = sum(out[s]["reps"][k]["r1"] * out[s]["n_q"] for s in seqs) / nq
            arm = float(np.mean([out[s]["reps"][k]["aliasrate_matched"] for s in seqs]))
            arf = float(np.mean([out[s]["reps"][k]["aliasrate_fixed"] for s in seqs]))
            summ[k] = {"r1_qw": r1, "aliasrate_matched_mean": arm,
                       "aliasrate_fixed_mean": arf}
        out[f"_summary_{tag}"] = summ
        print(f"== {tag} ==", flush=True)
        for k, v in summ.items():
            print(f"  {k:10} R@1={v['r1_qw']:.3f}  AR-matched={v['aliasrate_matched_mean']*100:.3f}%"
                  f"  AR-fixed={v['aliasrate_fixed_mean']*100:.3f}%", flush=True)

    json.dump(out, open(OUT / "aliasrate_size_sweep.json", "w"), indent=1)
    print("wrote", OUT / "aliasrate_size_sweep.json")


if __name__ == "__main__":
    main()
