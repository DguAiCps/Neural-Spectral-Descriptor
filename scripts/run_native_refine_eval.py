#!/usr/bin/env python3
"""B2-native evaluation: Bayesian-refined similarity edges from the pass-1 key.

Mirrors the trainer's two-pass refinement at eval time (per-sequence self-fit
approximation): fit a cosine SimilarityDistribution on the pass-1 re-metrized
416D embeddings against GT poses (5 m / 30-frame protocol, same as training
fit), rebuild similarity edges with it, forward again. Reported for both the
frozen seed checkpoints and the stage-5 retrained checkpoint so the retrain is
judged under its own graph-construction policy.
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
from keyframe.graph_manager import rebuild_similarity_edges
from utils.similarity_stats import SimilarityDistribution

CACHE = REPO / "data/preprocessed_cross_sensor_operating"
RVEC = REPO / "artifacts/key_remetrize_r.npy"
OUT = REPO / "results/_remetrize_twopass"
CKPTS = {
    "seed1": REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    "seed2": REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
    "seed3": REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
    "s5": REPO / "checkpoints/remetrize_twopass_s1/best_model.pth",
    "s5_2": REPO / "checkpoints/remetrize_twopass_s2/best_model.pth",
    "s5_3": REPO / "checkpoints/remetrize_twopass_s3/best_model.pth",
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
                with torch.no_grad():
                    embA = model_rm(graph.to(DEVICE)).cpu().numpy()

                dist = SimilarityDistribution(metric="cosine").fit(
                    _normalize(embA), poses, pos_dist=DTH, neg_dist=10.0,
                    min_temporal_gap=SKIP, n_samples=1_000_000)
                if not dist.fitted:
                    print(f"[{seed}] {name}: refit failed, skipping", flush=True)
                    continue
                rebuild_similarity_edges(
                    graph, _normalize(embA), similarity_dist=dist,
                    similarity_max_k=10, similarity_metric="cosine")
                with torch.no_grad():
                    embN = model_rm(graph.to(DEVICE)).cpu().numpy()
                report[name] = r1(embN, poses)
                print(f"[{seed}] {name:<20} B2native={report[name]:.3f}", flush=True)
        out = OUT / f"native_{seed}.json"
        json.dump(report, open(out, "w"), indent=1)
        print("saved", out, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["s5"])
