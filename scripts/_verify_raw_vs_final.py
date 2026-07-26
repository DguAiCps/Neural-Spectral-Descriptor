#!/usr/bin/env python3
"""Verify whether the 288D magnitude-only key and the 416D retrieval key give
identical KITTI Recall@1, and diagnose WHY (gate magnitude, top-1 flip count).

Reuses the exact retrieval primitives from evaluate_kitti_checkpoint.py so the
numbers are directly comparable to the paper tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import yaml
from evaluate_kitti_checkpoint import (
    _apply_encoder_preset,
    _make_model,
    _build_eval_graph,
    _cache_to_keyframes,
)
from run_kitti_operating_point import _find_queries, _topk_cosine, _score, _normalize

CACHE_DIR = REPO / "data/_verify_cache"
CKPTS = {
    "seed1": REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    "seed2": REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
    "seed3": REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
}
SEQS = ["00", "05", "08"]
DTH = 5.0
SKIP = 30
KVALS = [1, 5, 10]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def recall_and_ranked(embeddings, poses):
    normed = _normalize(embeddings)
    queries = _find_queries(poses, DTH, SKIP)
    ranked = [
        _topk_cosine(normed, q, max(KVALS) + 2 * SKIP, SKIP)[: max(KVALS)]
        for q, _ in queries
    ]
    metrics = _score(poses, queries, ranked, KVALS, DTH)
    top1 = np.array([r[0] for r in ranked])
    return metrics, queries, top1


def top1_correct(poses, queries, top1):
    out = []
    for (q, _), t in zip(queries, top1):
        out.append(np.linalg.norm(poses[q][:3, 3] - poses[int(t)][:3, 3]) < DTH)
    return np.array(out)


def main():
    cfg = _apply_encoder_preset(
        yaml.safe_load(open(REPO / "configs/training_multi_dataset.yaml")),
        "no_interdiff",
    )
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625

    report = {}
    for seed, ckpt in CKPTS.items():
        if not ckpt.exists():
            print(f"SKIP {seed}: missing {ckpt}")
            continue
        model = _make_model(cfg, ckpt, DEVICE)
        seed_rep = {}
        for seq in SEQS:
            cache = np.load(CACHE_DIR / f"kitti_operating_{seq}_layout60.npz")
            descriptors = cache["descriptors"].astype(np.float32)
            poses = cache["poses"]
            keyframes = _cache_to_keyframes(cache)
            graph = _build_eval_graph(
                keyframes=keyframes, poses=poses, descriptors=descriptors,
                cache=cache, config=cfg, device=DEVICE,
                temporal_edge_mode="bidirectional",
                temporal_direction_mode="none",
                similarity_min_k=0, phase_features=None, sensor_key="kitti",
            )
            with torch.no_grad():
                emb = model(graph.to(DEVICE)).detach().cpu().numpy()  # 416D

            raw = descriptors                     # 288D magnitude-only
            ctx = emb[:, 288:416]                 # gated context slice
            final = emb                           # 416D retrieval key

            raw_m, q_raw, top1_raw = recall_and_ranked(raw, poses)
            ctx_m, _, _ = recall_and_ranked(ctx, poses)
            fin_m, q_fin, top1_fin = recall_and_ranked(final, poses)

            # --- diagnostics the paper never reported ---
            # 1) is the first 288 dims of the 416D key == normalized raw?
            raw_norm = _normalize(raw)
            head_dev = float(np.abs(emb[:, :288] - raw_norm).max())
            # 2) effective per-node gate magnitude = norm of ctx slice of final
            ctx_norms = np.linalg.norm(emb[:, 288:416], axis=1)
            # 3) top-1 flips between raw and final
            flips = int(np.sum(top1_raw != top1_fin))
            n_q = len(top1_raw)
            corr_raw = top1_correct(poses, q_raw, top1_raw)
            corr_fin = top1_correct(poses, q_fin, top1_fin)
            gained = int(np.sum(~corr_raw & corr_fin))   # raw wrong -> final right
            lost = int(np.sum(corr_raw & ~corr_fin))     # raw right -> final wrong

            seed_rep[seq] = {
                "n_queries": n_q,
                "raw288_R@1": round(raw_m["R@1"], 6),
                "ctx128_R@1": round(ctx_m["R@1"], 6),
                "final416_R@1": round(fin_m["R@1"], 6),
                "head_max_abs_dev(288 of final vs norm raw)": head_dev,
                "ctx_slice_norm_mean": round(float(ctx_norms.mean()), 5),
                "ctx_slice_norm_min": round(float(ctx_norms.min()), 5),
                "ctx_slice_norm_max": round(float(ctx_norms.max()), 5),
                "top1_flips_raw_vs_final": flips,
                "top1_flips_gained(raw_wrong->final_right)": gained,
                "top1_flips_lost(raw_right->final_wrong)": lost,
            }
            print(f"[{seed} {seq}] raw288={raw_m['R@1']:.6f} "
                  f"ctx128={ctx_m['R@1']:.6f} final416={fin_m['R@1']:.6f} "
                  f"flips={flips} (+{gained}/-{lost}) "
                  f"ctxnorm~{ctx_norms.mean():.4f}", flush=True)
        report[seed] = seed_rep

    out = REPO / "results/_verify_raw_vs_final.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("WROTE", out)


if __name__ == "__main__":
    main()
