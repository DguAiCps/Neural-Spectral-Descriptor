#!/usr/bin/env python3
"""Recompute invariant-collision identifiability with all three deployed seeds.

Collision pairs and neighbourhood distances are fixed from the shared 288D
invariant block.  Only the context-dependent 416D separation is varied by
checkpoint seed, so the reported standard deviation is a matched-seed result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SEQUENCES = [
    "kitti_00", "kitti_05", "kitti_08", "nclt_2012-01-08",
    "nclt_2013-01-10", "helipr_Town01", "mulran_DCC03",
    "mulran_KAIST03", "mulran_Riverside03",
]
DUMP_DIRS = {
    1: REPO / "results/_final416_seed1",
    2: REPO / "results/_final416_seed2",
    3: REPO / "results/_final416_seed3",
}
CHECKPOINTS = {
    1: REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
    2: REPO / "checkpoints/800d_4sensor_seed2_20260604_125426/best_model.pth",
    3: REPO / "checkpoints/800d_4sensor_seed3_20260623_132834/best_model.pth",
}
EPS, GAMMA, POS_M, SKIP, WINDOW = 0.1, 25.0, 5.0, 30, 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def neighbourhoods(n: int) -> list[np.ndarray]:
    return [np.asarray([k for k in range(i - WINDOW, i + WINDOW + 1)
                        if k != i and 0 <= k < n], dtype=np.int64)
            for i in range(n)]


def chamfer(left: np.ndarray, right: np.ndarray) -> float:
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean()))


def collision_pairs(key: np.ndarray, poses: np.ndarray, rng: np.random.Generator,
                    samples: int, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    positions = poses[:, :3, 3]
    n = len(key)
    source = rng.integers(0, n, samples)
    target = rng.integers(0, n, samples)
    far = np.linalg.norm(positions[source] - positions[target], axis=1) >= GAMMA
    source, target = source[far], target[far]
    close = np.linalg.norm(key[source] - key[target], axis=1) <= EPS
    source, target = source[close], target[close]
    pairs = np.unique(np.minimum(source, target) * n + np.maximum(source, target))
    source, target = pairs // n, pairs % n
    if len(source) > maximum:
        selected = rng.choice(len(source), maximum, replace=False)
        source, target = source[selected], target[selected]
    return source.astype(np.int64), target.astype(np.int64)


def revisit_distances(key: np.ndarray, poses: np.ndarray, nbs: list[np.ndarray]) -> np.ndarray:
    positions = poses[:, :3, 3]
    pairs = []
    for query in range(SKIP, len(positions)):
        prior = np.linalg.norm(positions[:query - SKIP + 1] - positions[query], axis=1)
        match = np.flatnonzero(prior < POS_M)
        if len(match):
            pairs.append((query, int(match[0])))
    return np.asarray([chamfer(key[nbs[left]], key[nbs[right]]) for left, right in pairs])


def seed_metrics(embedding: np.ndarray, key: np.ndarray, context_distances: np.ndarray,
                 left: np.ndarray, right: np.ndarray, permutation_seed: int,
                 permutations: int) -> dict[str, float]:
    separation = np.linalg.norm(embedding[left] - embedding[right], axis=1)
    real_r = float(np.corrcoef(context_distances, separation)[0, 1])
    real_rate = float((separation > EPS).mean() * 100.0)
    context = embedding[:, 288:]
    rng = np.random.default_rng(permutation_seed)
    null_r, null_rate = [], []
    for _ in range(permutations):
        permuted = np.concatenate([key, context[rng.permutation(len(context))]], axis=1)
        values = np.linalg.norm(permuted[left] - permuted[right], axis=1)
        null_r.append(float(np.corrcoef(context_distances, values)[0, 1]))
        null_rate.append(float((values > EPS).mean() * 100.0))
    return {
        "pearson_r": real_r,
        "separation_rate_pct": real_rate,
        "permuted_pearson_r_mean": float(np.mean(null_r)),
        "permuted_pearson_r_max_abs": float(np.max(np.abs(null_r))),
        "permuted_separation_rate_pct_mean": float(np.mean(null_rate)),
    }


def mean_std(rows: dict[str, dict[str, float]], metric: str) -> dict[str, object]:
    values = [row[metric] for row in rows.values()]
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--samples", type=int, default=400000)
    parser.add_argument("--max-collisions", type=int, default=20000)
    parser.add_argument("--permutations", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/supplementary/identifiability_3seed.json")
    args = parser.parse_args()
    if set(args.seeds) - set(DUMP_DIRS):
        raise ValueError("available seeds are 1, 2, and 3")
    report = {
        "protocol": {
            "collision_definition": f"far pair >= {GAMMA:g}m with 288D invariant L2 distance <= {EPS:g}",
            "neighbourhood": f"symmetric Chamfer distance over fixed +/-{WINDOW} invariant-key neighbours",
            "separation": "L2 distance in the seed-specific 416D deployed retrieval key",
            "permutation": "permute only the 128D context block; stress test, not a dimension control",
            "seeds": args.seeds,
            "samples_per_sequence": args.samples,
            "max_collision_pairs": args.max_collisions,
            "permutations": args.permutations,
        },
        "provenance": {
            "command": shlex.join(__import__("sys").argv),
            "dump_dirs": {str(seed): str(DUMP_DIRS[seed].relative_to(REPO)) for seed in args.seeds},
            "checkpoints": {
                str(seed): {
                    "path": str(CHECKPOINTS[seed].relative_to(REPO)),
                    "sha256": sha256(CHECKPOINTS[seed]),
                } for seed in args.seeds
            },
            "checkpoint_note": "Seed-specific deployed 800D checkpoints; invariant 288D blocks are asserted identical.",
        },
        "sequences": {},
    }
    for sequence_index, sequence in enumerate(SEQUENCES):
        loaded = {seed: np.load(DUMP_DIRS[seed] / f"{sequence}.npz") for seed in args.seeds}
        poses = loaded[args.seeds[0]]["poses"]
        key = l2(loaded[args.seeds[0]]["emb416"][:, :288].astype(np.float64))
        for seed, item in loaded.items():
            if not np.array_equal(poses, item["poses"]):
                raise ValueError(f"pose mismatch: {sequence}/seed{seed}")
            if not np.array_equal(loaded[args.seeds[0]]["emb416"][:, :288], item["emb416"][:, :288]):
                raise ValueError(f"invariant-key mismatch: {sequence}/seed{seed}")
        left, right = collision_pairs(
            key, poses, np.random.default_rng(sequence_index), args.samples, args.max_collisions
        )
        source_hashes = {
            str(seed): sha256(DUMP_DIRS[seed] / f"{sequence}.npz") for seed in args.seeds
        }
        if len(left) < 20:
            report["sequences"][sequence] = {
                "n_keyframes": len(key),
                "n_collisions": int(len(left)),
                "source_sha256": source_hashes,
                "status": "not estimable under the fixed epsilon=0.1 collision protocol",
            }
            continue
        nbs = neighbourhoods(len(key))
        distances = np.asarray([chamfer(key[nbs[a]], key[nbs[b]]) for a, b in zip(left, right)])
        revisit = revisit_distances(key, poses, nbs)
        per_seed = {}
        for seed in args.seeds:
            embedding = loaded[seed]["emb416"].astype(np.float64)
            metrics = seed_metrics(
                embedding, key, distances, left, right,
                permutation_seed=10_000 * sequence_index + seed, permutations=args.permutations,
            )
            per_seed[str(seed)] = metrics
        report["sequences"][sequence] = {
            "n_keyframes": len(key),
            "n_collisions": int(len(left)),
            "n_revisit_pairs": int(len(revisit)),
            "neighbourhood_distance_median_collision": float(np.median(distances)),
            "neighbourhood_distance_median_revisit": float(np.median(revisit)),
            "per_seed": per_seed,
            "three_seed": {
                metric: mean_std(per_seed, metric)
                for metric in ("pearson_r", "separation_rate_pct", "permuted_pearson_r_mean", "permuted_separation_rate_pct_mean")
            },
            "source_sha256": source_hashes,
        }
        print(f"[{sequence}] collisions={len(left)} r="
              f"{report['sequences'][sequence]['three_seed']['pearson_r']['mean']:.3f}", flush=True)

    report["aggregate"] = {}
    aggregate_rows = {}
    for seed in args.seeds:
        sequence_rows = {
            name: row["per_seed"][str(seed)] for name, row in report["sequences"].items()
            if "per_seed" in row
        }
        if not sequence_rows:
            continue
        # Collision-pair indices are sequence-local, so aggregate metrics use
        # collision-count weighting over per-sequence values instead.
        weights = np.asarray([report["sequences"][name]["n_collisions"] for name in sequence_rows])
        aggregate_rows[str(seed)] = {
            metric: float(np.average([row[metric] for row in sequence_rows.values()], weights=weights))
            for metric in ("pearson_r", "separation_rate_pct", "permuted_pearson_r_mean", "permuted_separation_rate_pct_mean")
        }
        aggregate_rows[str(seed)]["n_collision_pairs"] = int(weights.sum())
    report["aggregate"] = {"per_seed_collision_weighted": aggregate_rows}
    if aggregate_rows:
        report["aggregate"]["three_seed"] = {
            metric: mean_std(aggregate_rows, metric)
            for metric in ("pearson_r", "separation_rate_pct", "permuted_pearson_r_mean", "permuted_separation_rate_pct_mean")
        }
    else:
        report["aggregate"]["note"] = (
            "No sequence reached the fixed minimum collision-pair count under epsilon=0.1; "
            "no threshold relaxation or seed statistic is reported."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
