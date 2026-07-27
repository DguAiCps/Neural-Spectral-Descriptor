#!/usr/bin/env python3
"""Table 32 (tab:role2x2) invariant rows: re-metrized 288D key, no context.

Retrieval key = normalize(d_raw * r) with the committed 288D re-metrization
vector (closed-form, seed-independent; artifacts/key_remetrize_r.npy). The
candidate/rerank protocol is identical to the deployed row: three-way union
pool at n_coarse=800, sketch_fft rerank grid, fusion_norm="raw".

  --split val    9 held-out sequences  -> results/_remetrize_twopass/role2x2_invariant_val.json
  --split train  23 training sequences -> results/_remetrize_twopass/role2x2_invariant_train.json
                 (per-sensor fusion-weight selection only, mirroring
                  calibrate_rerank_weights_train.py)

Run inside the pyg container:
  python scripts/_role2x2_invariant_eval.py --split val
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _recall_with_phase_sketch_fusion, _pool_rows
from run_kitti_operating_point import _find_queries, _topk_cosine, _score, _normalize

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
BEVDIR = REPO / "data/preprocessed_cross_sensor_bev_layout"
RVEC = REPO / "artifacts/key_remetrize_r.npy"
OUT = REPO / "results/_remetrize_twopass"
VAL = {
    "kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
    "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"],
}
TRAIN = {
    "kitti": ["01", "02", "06", "07"],
    "nclt": ["2012-05-11", "2012-08-04", "2012-11-04", "2012-11-16", "2013-02-23"],
    "helipr": ["Bridge01", "DCC04", "KAIST04", "Riverside04", "Roundabout01", "Town02"],
    "mulran": ["DCC01", "DCC02", "KAIST01", "KAIST02", "Riverside01", "Riverside02",
               "Sejong01", "Sejong02"],
}
VAL_BEV_DIRS = {"kitti": "preprocessed_kitti_bev_layout", "nclt": "preprocessed_nclt_bev_layout",
                "helipr": "preprocessed_helipr_bev_layout", "mulran": "preprocessed_mulran_bev_layout"}
DTH, SKIP = 5.0, 30


def r1(emb, poses):
    normed = _normalize(emb)
    q = _find_queries(poses, DTH, SKIP)
    ranked = [_topk_cosine(normed, i, 1 + 2 * SKIP, SKIP)[:1] for i, _ in q]
    return _score(poses, q, ranked, [1], DTH)["R@1"]


def load_bev(split, sensor, seq):
    if split == "train":
        f = BEVDIR / f"{sensor}_bev_layout_{seq}_s60_max_r1-80_z-3-5_h8.npz"
    else:
        stride = "_stride1" if sensor == "nclt" else ""
        f = REPO / "data" / VAL_BEV_DIRS[sensor] / \
            f"{sensor}_bev_layout_{seq}_s60_max_r1-80_z-3-5_h8{stride}.npz"
    return _pool_rows(np.load(f)["bev_layouts"].astype(np.float32), 16, "max")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "train"], default="val")
    args = ap.parse_args()
    seqset = VAL if args.split == "val" else TRAIN

    r = np.load(RVEC).astype(np.float32)
    report = {}
    for sensor, seqs in seqset.items():
        for seq in seqs:
            name = f"{sensor}/{seq}"
            cache = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
            d_rm = _normalize(cache["descriptors"].astype(np.float32) * r[None, :])
            poses = cache["poses"]
            coarse = r1(d_rm, poses)
            bev = load_bev(args.split, sensor, seq)
            res = _recall_with_phase_sketch_fusion(
                embeddings=d_rm, range_layouts=cache["nsd_layouts"].astype(np.float32),
                bev_layouts=bev, poses=poses, k_values=[1], distance_threshold=DTH,
                skip_frames=SKIP, n_coarse=800, range_freqs=4, bev_freqs=8, n_sectors=60,
                sketch_bev_weights=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
                sketch_range_weights=[0.0, 0.5, 1.0, 2.0, 4.0],
                rerank_mode="sketch_fft", fusion_norm="raw")
            grid = {k: v["R@1"] for k, v in res.items() if not k.startswith("_")}
            report[name] = {"coarse": coarse, "grid": grid}
            print(f"[inv/{args.split}] {name:<22} coarse={coarse:.4f} "
                  f"bev0.5={grid['phase_sketch_fusion_bev0.5_range0']:.4f} "
                  f"best={max(grid.values()):.4f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"role2x2_invariant_{args.split}.json"
    json.dump(report, open(out, "w"), indent=1)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
