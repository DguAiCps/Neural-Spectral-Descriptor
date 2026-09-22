#!/usr/bin/env python3
"""Measure sensitivity to noise injected only into temporal edge attributes.

Keyframes, invariant descriptors, graph connectivity, similarity-edge attributes,
queries, and retrieval ground truth remain fixed.  Independent Gaussian position
and yaw perturbations alter only the temporal distance/rotation edge attributes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
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
from run_kitti_operating_point import _find_queries, _normalize, _topk_cosine  # noqa: E402


CACHE = REPO / "data/preprocessed_cross_sensor_operating"
CHECKPOINT = REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth"
CONFIG = REPO / "configs/training_multi_dataset.yaml"
DTH, SKIP = 5.0, 30
SEQUENCES = [
    ("kitti", "00"), ("kitti", "05"), ("kitti", "08"),
    ("nclt", "2012-01-08"), ("nclt", "2013-01-10"),
    ("helipr", "Town01"),
]
CONDITIONS = {
    "clean": (0.0, 0.0),
    "translation_0p25m": (0.25, 0.0),
    "translation_1m": (1.0, 0.0),
    "yaw_2deg": (0.0, 2.0),
    "yaw_10deg": (0.0, 10.0),
    "joint_1m_10deg": (1.0, 10.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_recall_at_1(embedding: np.ndarray, clean_poses: np.ndarray) -> tuple[float, int]:
    key = _normalize(embedding)
    positions = clean_poses[:, :3, 3]
    queries = _find_queries(clean_poses, DTH, SKIP)
    hits = 0
    for query, _ in queries:
        ranked = _topk_cosine(key, query, 1 + 2 * SKIP, SKIP)
        hits += int(len(ranked) and np.linalg.norm(positions[ranked[0]] - positions[query]) < DTH)
    return hits / max(len(queries), 1), len(queries)


def perturbed_poses(clean_poses: np.ndarray, translation_sigma_m: float,
                    yaw_sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    noisy = clean_poses.astype(np.float64, copy=True)
    if translation_sigma_m:
        noisy[:, :3, 3] += rng.normal(0.0, translation_sigma_m, size=(len(noisy), 3))
    if yaw_sigma_deg:
        angles = np.deg2rad(rng.normal(0.0, yaw_sigma_deg, size=len(noisy)))
        c, s = np.cos(angles), np.sin(angles)
        rotations = np.zeros((len(noisy), 3, 3), dtype=np.float64)
        rotations[:, 0, 0], rotations[:, 0, 1] = c, -s
        rotations[:, 1, 0], rotations[:, 1, 1] = s, c
        rotations[:, 2, 2] = 1.0
        noisy[:, :3, :3] = rotations @ noisy[:, :3, :3]
    return noisy


def temporal_attribute_noise(graph, clean_poses: np.ndarray,
                             translation_sigma_m: float, yaw_sigma_deg: float,
                             rng: np.random.Generator):
    """Clone a graph and alter only [distance, rotation] on temporal edges."""
    noisy_poses = perturbed_poses(clean_poses, translation_sigma_m, yaw_sigma_deg, rng)
    noisy_graph = graph.clone()
    noisy_graph.edge_attr = graph.edge_attr.clone()
    temporal_mask = graph.edge_type == 0
    src, dst = graph.edge_index[:, temporal_mask].detach().cpu().numpy()
    delta = noisy_poses[src, :3, 3] - noisy_poses[dst, :3, 3]
    distances = np.linalg.norm(delta, axis=1)
    rel_rotations = noisy_poses[dst, :3, :3] @ np.swapaxes(noisy_poses[src, :3, :3], 1, 2)
    traces = np.trace(rel_rotations, axis1=1, axis2=2)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    replacement = torch.from_numpy(np.stack([
        np.log1p(distances) / 5.0,
        angles / np.pi,
    ], axis=1).astype(np.float32)).to(noisy_graph.edge_attr.device)
    noisy_graph.edge_attr[temporal_mask, :2] = replacement

    if not torch.equal(graph.edge_index, noisy_graph.edge_index):
        raise AssertionError("pose perturbation changed graph connectivity")
    if not torch.equal(graph.edge_type, noisy_graph.edge_type):
        raise AssertionError("pose perturbation changed edge types")
    similarity_mask = ~temporal_mask
    if not torch.equal(graph.edge_attr[similarity_mask], noisy_graph.edge_attr[similarity_mask]):
        raise AssertionError("pose perturbation changed similarity-edge attributes")
    changed = (graph.edge_attr[temporal_mask, :2] - noisy_graph.edge_attr[temporal_mask, :2]).abs()
    audit = {
        "temporal_edges": int(temporal_mask.sum().item()),
        "similarity_edges": int(similarity_mask.sum().item()),
        "max_abs_temporal_attribute_change": float(changed.max().item()) if len(changed) else 0.0,
        "mean_abs_temporal_attribute_change": float(changed.mean().item()) if len(changed) else 0.0,
        "connectivity_unchanged": True,
        "similarity_attributes_unchanged": True,
    }
    return noisy_graph, audit


def load_model(device: str):
    cfg = _apply_encoder_preset(yaml.safe_load(CONFIG.read_text()), "no_interdiff")
    cfg["gnn"]["use_residual_gate"] = True
    cfg["gnn"]["gate_initial_alpha"] = 0.0625
    return cfg, _make_model(cfg, CHECKPOINT, device)


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n_noise_seeds": len(values),
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS),
                        default=list(CONDITIONS))
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sequences", nargs="+", default=[f"{s}/{q}" for s, q in SEQUENCES])
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/supplementary/pose_noise_sensitivity.json")
    args = parser.parse_args()
    selected = []
    allowed = {f"{sensor}/{sequence}": (sensor, sequence) for sensor, sequence in SEQUENCES}
    for item in args.sequences:
        if item not in allowed:
            raise ValueError(f"sequence must be one of {sorted(allowed)}, got {item}")
        selected.append((item, *allowed[item]))

    cfg, model = load_model(args.device)
    report = {
        "protocol": {
            "keyframes": "fixed cache keyframes",
            "retrieval_ground_truth": "clean cached poses; <5 m revisit after a 30-frame skip",
            "noise_scope": "only temporal edge_attr[:, :2] (relative distance and rotation)",
            "noise_distribution": "independent per-keyframe zero-mean Gaussian translation and yaw",
            "conditions": {
                name: {"translation_sigma_m": values[0], "yaw_sigma_deg": values[1]}
                for name, values in CONDITIONS.items() if name in args.conditions
            },
            "noise_seeds": args.noise_seeds,
            "sequences": [item for item, _, _ in selected],
            "distance_threshold_m": DTH,
            "skip_frames": SKIP,
        },
        "provenance": {
            "command": shlex.join(sys.argv),
            "config": str(CONFIG.relative_to(REPO)),
            "config_sha256": sha256(CONFIG),
            "checkpoint": str(CHECKPOINT.relative_to(REPO)),
            "checkpoint_sha256": sha256(CHECKPOINT),
        },
        "conditions": {},
    }
    loaded = {}
    for sequence_key, sensor, sequence in selected:
        cache_path = CACHE / f"{sensor}_operating_{sequence}_layout60_stride1.npz"
        cache = np.load(cache_path)
        graph = _build_eval_graph(
            keyframes=_cache_to_keyframes(cache), poses=cache["poses"],
            descriptors=cache["descriptors"].astype(np.float32), cache=cache,
            config=cfg, device=args.device, temporal_edge_mode="bidirectional",
            temporal_direction_mode="none", similarity_min_k=0, phase_features=None,
            sensor_key=sensor,
        )
        loaded[sequence_key] = (cache, graph, cache_path)

    for condition in args.conditions:
        translation_sigma_m, yaw_sigma_deg = CONDITIONS[condition]
        per_seed = {}
        seeds = [0] if condition == "clean" else args.noise_seeds
        for noise_seed in seeds:
            per_sequence, weighted_sum, total_queries = {}, 0.0, 0
            for sequence_index, (sequence_key, _, _) in enumerate(selected):
                cache, graph, cache_path = loaded[sequence_key]
                rng = np.random.default_rng(1_000_003 * noise_seed + sequence_index)
                if condition == "clean":
                    used_graph = graph
                    audit = {
                        "temporal_edges": int((graph.edge_type == 0).sum().item()),
                        "similarity_edges": int((graph.edge_type == 1).sum().item()),
                        "max_abs_temporal_attribute_change": 0.0,
                        "mean_abs_temporal_attribute_change": 0.0,
                        "connectivity_unchanged": True,
                        "similarity_attributes_unchanged": True,
                    }
                else:
                    used_graph, audit = temporal_attribute_noise(
                        graph, cache["poses"], translation_sigma_m, yaw_sigma_deg, rng
                    )
                with torch.no_grad():
                    embedding = model(used_graph.to(args.device)).detach().cpu().numpy()
                recall, n_queries = raw_recall_at_1(embedding, cache["poses"])
                per_sequence[sequence_key] = {
                    "recall_at_1": recall,
                    "n_queries": n_queries,
                    "cache_path": str(cache_path.relative_to(REPO)),
                    "cache_sha256": sha256(cache_path),
                    "edge_attribute_audit": audit,
                }
                weighted_sum += recall * n_queries
                total_queries += n_queries
                print(f"[{condition} seed={noise_seed}] {sequence_key}: {recall:.4f}", flush=True)
            per_seed[str(noise_seed)] = {
                "query_weighted_recall_at_1": weighted_sum / total_queries,
                "n_queries_total": total_queries,
                "per_sequence": per_sequence,
            }
        report["conditions"][condition] = {
            "per_noise_seed": per_seed,
            "query_weighted_recall_at_1": summarize([
                row["query_weighted_recall_at_1"] for row in per_seed.values()
            ]),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
