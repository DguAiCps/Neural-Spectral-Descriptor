#!/usr/bin/env python3
"""Diagnose phase-neighbor graph operating points from a cached dataset."""

import argparse
from pathlib import Path

import numpy as np


def _normalize_phase(phase: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float32)
    phase = np.sign(phase) * np.log1p(np.abs(phase))
    phase = phase - phase.mean(axis=0, keepdims=True)
    phase = phase / np.maximum(phase.std(axis=0, keepdims=True), 1e-6)
    phase = phase / np.maximum(np.linalg.norm(phase, axis=1, keepdims=True), 1e-8)
    return phase.astype(np.float32)


def _search(phase: np.ndarray, fetch_k: int):
    try:
        import faiss

        index = faiss.IndexFlatIP(phase.shape[1])
        index.add(phase)
        return index.search(phase, fetch_k)
    except Exception:
        sims = phase @ phase.T
        indices = np.argsort(-sims, axis=1)[:, :fetch_k]
        return np.take_along_axis(sims, indices, axis=1), indices


def diagnose(
    cache_path: Path,
    max_ks: list[int],
    thresholds: list[float],
    temporal_exclude: int,
    positive_distance: float,
) -> None:
    data = np.load(cache_path)
    if "phase_features" not in data:
        raise KeyError(f"{cache_path} has no phase_features array")
    if "poses" not in data:
        raise KeyError(f"{cache_path} has no poses array")

    phase = _normalize_phase(data["phase_features"])
    poses = data["poses"]
    positions = poses[:, :3, 3].astype(np.float32)
    sequence_ids = data["sequence_ids"] if "sequence_ids" in data else None
    n_nodes = phase.shape[0]
    fetch_k = min(n_nodes, max(max(max_ks) * 8 + temporal_exclude + 1, max(max_ks) + 1))
    sims, indices = _search(phase, fetch_k)

    print(f"cache={cache_path}")
    print(f"nodes={n_nodes} phase_dim={phase.shape[1]} fetch_k={fetch_k}")
    print("max_k\tthreshold\tedges\tpos_rate\tmean_sim\tmedian_sim")
    for max_k in max_ks:
        for threshold in thresholds:
            labels = []
            kept_sims = []
            for i in range(n_nodes):
                added = 0
                for sim, j in zip(sims[i], indices[i]):
                    j = int(j)
                    if i == j:
                        continue
                    same_sequence = sequence_ids is None or sequence_ids[i] == sequence_ids[j]
                    if not same_sequence:
                        continue
                    if temporal_exclude > 0 and same_sequence and abs(i - j) <= temporal_exclude:
                        continue
                    if float(sim) < threshold:
                        continue
                    dist = float(np.linalg.norm(positions[i] - positions[j]))
                    labels.append(1.0 if dist < positive_distance else 0.0)
                    kept_sims.append(float(sim))
                    added += 1
                    if added >= max_k:
                        break
            if labels:
                labels_arr = np.asarray(labels, dtype=np.float32)
                sims_arr = np.asarray(kept_sims, dtype=np.float32)
                print(
                    f"{max_k}\t{threshold:.3f}\t{len(labels)}\t"
                    f"{labels_arr.mean():.3f}\t{sims_arr.mean():.3f}\t"
                    f"{np.median(sims_arr):.3f}"
                )
            else:
                print(f"{max_k}\t{threshold:.3f}\t0\t0.000\t0.000\t0.000")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_path", type=Path)
    parser.add_argument("--max-ks", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )
    parser.add_argument("--temporal-exclude", type=int, default=30)
    parser.add_argument("--positive-distance", type=float, default=5.0)
    args = parser.parse_args()
    diagnose(
        args.cache_path,
        args.max_ks,
        args.thresholds,
        args.temporal_exclude,
        args.positive_distance,
    )


if __name__ == "__main__":
    main()
