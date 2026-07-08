#!/usr/bin/env python3
"""Full-pipeline (phase-rerank included) evaluation of the A/B2 variants.

Runs the closed-form cylindrical+BEV phase-sketch rerank (range 4 freqs +
BEV 8 freqs on 16x60 layouts, top-800 pool, sketch_fft scoring) on top of:
  P   baseline 416D coarse key (frozen)
  A   re-metrized coarse key (fixed diag_wccn)
  B2  two-pass coarse key (similarity edges rebuilt from the pass-1 A key)

The full (w_bev, w_range) fusion-weight grid is evaluated in one pass for
every variant; comparisons should be read at a single weight pair fixed
across variants (anchored by whichever pair best reproduces the paper's
baseline row).
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import (
    _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes,
    _recall_with_phase_sketch_fusion, _pool_rows,
)
from run_kitti_operating_point import _find_queries, _normalize

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
BEVDIR = {
    "kitti": REPO / "data/preprocessed_kitti_bev_layout",
    "nclt": REPO / "data/preprocessed_nclt_bev_layout",
    "helipr": REPO / "data/preprocessed_helipr_bev_layout",
    "mulran": REPO / "data/preprocessed_mulran_bev_layout",
}
RVEC = REPO / "artifacts/key_remetrize_r.npy"
OUT = REPO / "results/_remetrize_twopass"
CKPTS = {
    "seed1": REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    "seed2": REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
    "seed3": REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
}
SENSORS = {
    "kitti": ["00", "05", "08"], "nclt": ["2012-01-08", "2013-01-10"],
    "helipr": ["Town01"], "mulran": ["DCC03", "KAIST03", "Riverside03"],
}
DTH, SKIP = 5.0, 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRIDS = {
    "minmax": ([0.5, 1.0, 2.0, 4.0], [0.0, 0.5, 1.0, 2.0]),
    "raw": ([0.0, 0.5, 1.0, 2.0, 4.0, 8.0], [0.0, 0.5, 1.0, 2.0, 4.0]),
}


def bev_path(sensor, seq):
    stride = "_stride1" if sensor == "nclt" else ""
    return BEVDIR[sensor] / f"{sensor}_bev_layout_{seq}_s60_max_r1-80_z-3-5_h8{stride}.npz"


def main(seeds, fusion="minmax"):
    W_BEV, W_RANGE = GRIDS[fusion]
    OUT.mkdir(parents=True, exist_ok=True)
    base_cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    base_cfg["gnn"]["use_residual_gate"] = True
    base_cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(base_cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(RVEC)}
    r = np.load(RVEC).astype(np.float32)

    for seed in seeds:
        model_base = _make_model(base_cfg, CKPTS[seed], DEVICE)
        model_rm = _make_model(rm_cfg, CKPTS[seed], DEVICE)
        report = {}
        for sensor, seqs in SENSORS.items():
            for seq in seqs:
                name = f"{sensor}/{seq}"
                cache = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
                d_raw = cache["descriptors"].astype(np.float32)
                poses = cache["poses"]
                rng_layouts = cache["nsd_layouts"].astype(np.float32)
                bev = _pool_rows(np.load(bev_path(sensor, seq))["bev_layouts"].astype(np.float32),
                                 16, "max")

                g_paper = _build_eval_graph(
                    keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
                    cache=cache, config=base_cfg, device=DEVICE,
                    temporal_edge_mode="bidirectional", temporal_direction_mode="none",
                    similarity_min_k=0, phase_features=None, sensor_key=sensor)
                with torch.no_grad():
                    embP = model_base(g_paper.to(DEVICE)).cpu().numpy()
                    embA = model_rm(g_paper.to(DEVICE)).cpu().numpy()

                cfg_b2 = copy.deepcopy(rm_cfg)
                cfg_b2["keyframe"].setdefault("graph", {})["similarity_threshold"] = -1.0
                g_b2 = _build_eval_graph(
                    keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=embA,
                    cache=cache, config=cfg_b2, device=DEVICE,
                    temporal_edge_mode="bidirectional", temporal_direction_mode="none",
                    similarity_min_k=0, phase_features=None, sensor_key=sensor)
                with torch.no_grad():
                    embB2 = model_rm(g_b2.to(DEVICE)).cpu().numpy()

                report[name] = {}
                for variant, emb in (("P", embP), ("A", embA), ("B2", embB2)):
                    res = _recall_with_phase_sketch_fusion(
                        embeddings=emb, range_layouts=rng_layouts, bev_layouts=bev,
                        poses=poses, k_values=[1], distance_threshold=DTH,
                        skip_frames=SKIP, n_coarse=800, range_freqs=4, bev_freqs=8,
                        n_sectors=60, sketch_bev_weights=W_BEV,
                        sketch_range_weights=W_RANGE, rerank_mode="sketch_fft",
                        fusion_norm=fusion)
                    report[name][variant] = {
                        k: v["R@1"] for k, v in res.items() if not k.startswith("_")}
                nq = len(_find_queries(poses, DTH, SKIP))
                report[name]["n_q"] = nq
                best = {v: max(report[name][v].values()) for v in ("P", "A", "B2")}
                print(f"[{seed}] {name:<20} best-grid " + "  ".join(
                    f"{v}={best[v]:.3f}" for v in ("P", "A", "B2")), flush=True)
        suffix = "" if fusion == "minmax" else f"_{fusion}"
        out = OUT / f"rerank{suffix}_{seed}.json"
        json.dump(report, open(out, "w"), indent=1)
        print("saved", out, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    fusion = "minmax"
    if args and args[0] in GRIDS:
        fusion, args = args[0], args[1:]
    main(args or ["seed1"], fusion)
