#!/usr/bin/env python3
"""Re-evaluate the NCLT rows of the context-isolation table.

This targets the current cross-sensor operating caches that reproduce the
canonical raw-288D Recall@1 values (0.298/0.166). It evaluates the three frozen
refiner checkpoints on CPU by default and also regenerates the ten-draw
within-sequence context-permutation control.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_kitti_checkpoint import (  # noqa: E402
    _apply_encoder_preset,
    _build_eval_graph,
    _cache_to_keyframes,
    _make_model,
)

CACHE_DIR = REPO / "data/preprocessed_cross_sensor_operating"
CACHES = {
    "nclt/2012-01-08": CACHE_DIR / "nclt_operating_2012-01-08_layout60_stride1.npz",
    "nclt/2013-01-10": CACHE_DIR / "nclt_operating_2013-01-10_layout60_stride1.npz",
}
CHECKPOINTS = {
    "seed1": REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    "seed2": REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
    "seed3": REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
}
DISTANCE_THRESHOLD = 5.0
SKIP_FRAMES = 30
N_PERMUTATIONS = 10


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def find_queries(poses: np.ndarray) -> np.ndarray:
    positions = poses[:, :3, 3]
    queries = []
    for query_idx in range(SKIP_FRAMES, len(positions)):
        prior = positions[: query_idx - SKIP_FRAMES + 1]
        if np.any(np.linalg.norm(prior - positions[query_idx], axis=1) < DISTANCE_THRESHOLD):
            queries.append(query_idx)
    return np.asarray(queries, dtype=np.int64)


def recall_at_one(keys: np.ndarray, poses: np.ndarray, queries: np.ndarray) -> float:
    keys = normalize(keys)
    positions = poses[:, :3, 3]
    all_indices = np.arange(len(keys))
    correct = 0
    block_size = 128
    for start in range(0, len(queries), block_size):
        q = queries[start : start + block_size]
        similarities = keys[q] @ keys.T
        invalid = np.abs(all_indices[None, :] - q[:, None]) < SKIP_FRAMES
        similarities[invalid] = -np.inf
        matches = np.argmax(similarities, axis=1)
        correct += int(
            np.sum(np.linalg.norm(positions[matches] - positions[q], axis=1) < DISTANCE_THRESHOLD)
        )
    return correct / max(len(queries), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/_verify_context_nclt_canonical.json",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cfg = _apply_encoder_preset(
        yaml.safe_load((REPO / "configs/training_multi_dataset.yaml").read_text()),
        "no_interdiff",
    )
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625

    report = {
        "protocol": {
            "device": args.device,
            "threads": args.threads,
            "distance_threshold_m": DISTANCE_THRESHOLD,
            "skip_frames": SKIP_FRAMES,
            "permutations_per_seed": N_PERMUTATIONS,
            "caches": {key: str(path.relative_to(REPO)) for key, path in CACHES.items()},
        },
        "seeds": {},
    }

    for seed_index, (seed, checkpoint) in enumerate(CHECKPOINTS.items(), start=1):
        model = _make_model(cfg, checkpoint, args.device)
        seed_report = {}
        for sequence, cache_path in CACHES.items():
            cache = np.load(cache_path)
            descriptors = cache["descriptors"].astype(np.float32)
            poses = cache["poses"]
            queries = find_queries(poses)
            graph = _build_eval_graph(
                keyframes=_cache_to_keyframes(cache),
                poses=poses,
                descriptors=descriptors,
                cache=cache,
                config=cfg,
                device=args.device,
                temporal_edge_mode="bidirectional",
                temporal_direction_mode="none",
                similarity_min_k=0,
                phase_features=None,
                sensor_key="nclt",
            )
            with torch.no_grad():
                embeddings = model(graph.to(args.device)).detach().cpu().numpy()

            raw_r1 = recall_at_one(descriptors, poses, queries)
            context_r1 = recall_at_one(embeddings[:, 288:416], poses, queries)
            final_r1 = recall_at_one(embeddings, poses, queries)

            rng = np.random.default_rng(1000 + seed_index)
            permutation_r1 = []
            for _ in range(N_PERMUTATIONS):
                permuted = embeddings.copy()
                order = rng.permutation(len(permuted))
                permuted[:, 288:416] = permuted[order, 288:416]
                permutation_r1.append(recall_at_one(permuted, poses, queries))

            seed_report[sequence] = {
                "n_queries": int(len(queries)),
                "raw288": raw_r1,
                "context128": context_r1,
                "final416": final_r1,
                "delta": final_r1 - raw_r1,
                "permuted416": permutation_r1,
            }
            print(
                f"[{seed} {sequence}] q={len(queries)} raw={raw_r1:.4f} "
                f"context={context_r1:.4f} final={final_r1:.4f} "
                f"perm={np.mean(permutation_r1):.4f}",
                flush=True,
            )
        report["seeds"][seed] = seed_report
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))

    aggregate = {}
    for sequence in CACHES:
        rows = [report["seeds"][seed][sequence] for seed in CHECKPOINTS]
        aggregate[sequence] = {
            "n_queries": rows[0]["n_queries"],
            "raw288": float(np.mean([row["raw288"] for row in rows])),
            "context128": float(np.mean([row["context128"] for row in rows])),
            "final416": float(np.mean([row["final416"] for row in rows])),
            "delta": float(np.mean([row["delta"] for row in rows])),
            "permuted416": float(
                np.mean([value for row in rows for value in row["permuted416"]])
            ),
        }
    report["three_seed_means"] = aggregate
    args.output.write_text(json.dumps(report, indent=2))
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
