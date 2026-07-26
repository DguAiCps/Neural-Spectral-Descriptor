#!/usr/bin/env python3
"""Same-place spread analysis: the false-negative failure mode AliasRate is blind to.

For every genuine revisit query (a keyframe with a true match >=30 frames earlier,
within 5 m) we measure, for the 288D magnitude key and the 416D retrieval key:
  - best same-place descriptor distance (nearest true positive)  -> "spread"
  - nearest-distractor descriptor distance                       -> retrieval competitor
  - SpreadRate = Pr(best same-place distance > eps) | genuine revisit
  - failure decomposition: R@1 misses that are spread-dominated (true match beyond
    eps) vs alias/near-field (a closer wrong place outranks a close true match)
  - forward (|dyaw|<=30) vs reverse (|dyaw|>90) same-place spread
Raw 288D is seed-independent; the 416D key is averaged over the 3 main checkpoints.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
CKPTS = {
    "seed1": REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    "seed2": REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
    "seed3": REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
}
SENSORS = {"kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
           "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"]}
DTH, SKIP, EPS, GAMMA = 5.0, 30, 0.1, 25.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def norm(X):
    X = X.astype(np.float64)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def yaw_of(poses):
    return np.arctan2(poses[:, 1, 0], poses[:, 0, 0])


def analyze(r, poses):
    """Return per-query arrays for one descriptor set."""
    pos = poses[:, :3, 3]; yaw = yaw_of(poses); n = len(r)
    best_pos, near_dis, dyaw, dis_geo, valid = [], [], [], [], []
    for q in range(n):
        mask = np.abs(np.arange(n) - q) >= SKIP
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        geo = np.linalg.norm(pos[idx] - pos[q], axis=1)
        posi = idx[geo < DTH]
        dist = idx[geo >= DTH]
        if posi.size == 0:
            continue  # not a revisit query
        dd_pos = np.linalg.norm(r[posi] - r[q], axis=1)
        bp = dd_pos.min(); bp_idx = posi[dd_pos.argmin()]
        best_pos.append(bp)
        # yaw diff to the nearest-in-pose true positive
        gp = posi[np.argmin(np.linalg.norm(pos[posi] - pos[q], axis=1))]
        dyaw.append(np.degrees(np.abs(np.arctan2(np.sin(yaw[q] - yaw[gp]), np.cos(yaw[q] - yaw[gp])))))
        if dist.size > 0:
            dd_dis = np.linalg.norm(r[dist] - r[q], axis=1)
            nd = dd_dis.min(); nd_idx = dist[dd_dis.argmin()]
            near_dis.append(nd)
            dis_geo.append(np.linalg.norm(pos[nd_idx] - pos[q]))
        else:
            near_dis.append(np.inf); dis_geo.append(np.inf)
        valid.append(q)
    return (np.array(best_pos), np.array(near_dis), np.array(dyaw), np.array(dis_geo))


def summarize(bp, nd, dyaw, dgeo):
    miss = bp >= nd  # R@1 wrong: a distractor is closer than the best true positive
    n = len(bp); nmiss = int(miss.sum())
    spread_dom = miss & (bp > EPS)          # true match itself beyond eps
    alias_dom = miss & (bp <= EPS) & (dgeo >= GAMMA)  # close true match outranked by a FAR place
    nearfield = miss & (bp <= EPS) & (dgeo < GAMMA)   # outranked by a near (5-25m) place
    fwd = dyaw <= 30.0; rev = dyaw > 90.0
    def med(x): return float(np.median(x)) if len(x) else float("nan")
    return {
        "n_query": n, "R@1": round(1 - nmiss / max(n, 1), 4),
        "sameplace_med": round(med(bp), 4),
        "SpreadRate_%": round(100 * float(np.mean(bp > EPS)), 2),
        "near_distractor_med": round(med(nd), 4),
        "miss_%": round(100 * nmiss / max(n, 1), 2),
        "of_miss_spread_%": round(100 * int(spread_dom.sum()) / max(nmiss, 1), 1),
        "of_miss_alias_%": round(100 * int(alias_dom.sum()) / max(nmiss, 1), 1),
        "of_miss_nearfield_%": round(100 * int(nearfield.sum()) / max(nmiss, 1), 1),
        "sameplace_med_fwd": round(med(bp[fwd]), 4),
        "sameplace_med_rev": round(med(bp[rev]), 4),
    }


def main():
    cfg = _apply_encoder_preset(yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True; cfg["gnn"]["gate_initial_alpha"] = 0.0625
    models = {s: _make_model(cfg, c, DEVICE) for s, c in CKPTS.items() if c.exists()}

    report = {}
    for sensor, seqs in SENSORS.items():
        for seq in seqs:
            f = CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz"
            if not f.exists():
                continue
            cache = np.load(f); desc = cache["descriptors"].astype(np.float32); poses = cache["poses"]
            # raw 288D (seed-independent)
            raw = summarize(*analyze(norm(desc), poses))
            # 416D key averaged over seeds -> summarize per seed then average scalars
            per_seed = []
            for s, model in models.items():
                graph = _build_eval_graph(
                    keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=desc, cache=cache,
                    config=cfg, device=DEVICE, temporal_edge_mode="bidirectional",
                    temporal_direction_mode="none", similarity_min_k=0, phase_features=None, sensor_key=sensor)
                with torch.no_grad():
                    emb = model(graph.to(DEVICE)).detach().cpu().numpy()
                per_seed.append(summarize(*analyze(norm(emb), poses)))
            keys = per_seed[0].keys()
            fin = {k: round(float(np.mean([ps[k] for ps in per_seed])), 4) for k in keys}
            report[f"{sensor}/{seq}"] = {"raw288": raw, "key416": fin}
            print(f"{sensor}/{seq:<12} "
                  f"spread(raw/key)={raw['sameplace_med']:.3f}/{fin['sameplace_med']:.3f} "
                  f"SpreadRate={raw['SpreadRate_%']:.1f}/{fin['SpreadRate_%']:.1f}% "
                  f"R@1={raw['R@1']:.3f}/{fin['R@1']:.3f} "
                  f"miss:spread{raw['of_miss_spread_%']:.0f}%/alias{raw['of_miss_alias_%']:.0f}%", flush=True)
    (REPO / "results/_verify_samespread.json").write_text(json.dumps(report, indent=2))
    print("WROTE results/_verify_samespread.json")


if __name__ == "__main__":
    main()
