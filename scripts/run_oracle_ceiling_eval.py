#!/usr/bin/env python3
"""Graph-topology oracle ceiling for the re-metrized pipeline.

Replaces similarity edges with ground-truth pose edges (5 m, >=30-frame gap,
top-10) on top of the preserved temporal edges, then forwards the frozen
800D checkpoint with the fixed diag_wccn key re-metrization (A). The gap
between this ceiling and B2 quantifies how much of the remaining error is
still graph-topology, per sensor.
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate_kitti_checkpoint import _apply_encoder_preset, _make_model, _build_eval_graph, _cache_to_keyframes
from run_kitti_operating_point import _find_queries, _topk_cosine, _score, _normalize
from keyframe.graph_manager import attach_pose_gt_similarity_edges

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
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


def r1(emb, poses):
    normed = _normalize(emb)
    q = _find_queries(poses, DTH, SKIP)
    ranked = [_topk_cosine(normed, i, 1 + 2 * SKIP, SKIP)[:1] for i, _ in q]
    return _score(poses, q, ranked, [1], DTH)["R@1"]


def main(seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    base_cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")), "no_interdiff")
    base_cfg["gnn"]["use_residual_gate"] = True
    base_cfg["gnn"]["gate_initial_alpha"] = 0.0625
    rm_cfg = copy.deepcopy(base_cfg)
    rm_cfg["gnn"]["key_remetrize"] = {"enabled": True, "init_path": str(RVEC)}

    for seed in seeds:
        model_rm = _make_model(rm_cfg, CKPTS[seed], DEVICE)
        report = {}
        for sensor, seqs in SENSORS.items():
            for seq in seqs:
                name = f"{sensor}/{seq}"
                cache = np.load(CACHE / f"{sensor}_operating_{seq}_layout60_stride1.npz")
                d_raw = cache["descriptors"].astype(np.float32)
                poses = cache["poses"]
                graph = _build_eval_graph(
                    keyframes=_cache_to_keyframes(cache), poses=poses, descriptors=d_raw,
                    cache=cache, config=rm_cfg, device=DEVICE,
                    temporal_edge_mode="bidirectional", temporal_direction_mode="none",
                    similarity_min_k=0, phase_features=None, sensor_key=sensor)
                graph, n_sim = attach_pose_gt_similarity_edges(
                    graph, poses, d_raw, pos_dist=DTH, min_temporal_gap=SKIP,
                    similarity_max_k=10)
                with torch.no_grad():
                    emb = model_rm(graph.to(DEVICE)).cpu().numpy()
                report[name] = {"oracle_A": r1(emb, poses), "n_sim_edges": int(n_sim)}
                print(f"[{seed}] {name:<20} oracle_A={report[name]['oracle_A']:.3f} "
                      f"(sim_edges={n_sim})", flush=True)
        out = OUT / f"oracle_{seed}.json"
        json.dump(report, open(out, "w"), indent=1)
        print("saved", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["seed1"])
