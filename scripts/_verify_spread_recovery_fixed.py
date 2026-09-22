#!/usr/bin/env python3
"""Measure context recovery on a fixed set of invariant-key spread misses.

The subset is defined once in the 288D invariant-key space: a headline query is
included when raw cosine retrieval misses and its best true-match L2 distance
exceeds EPS. The same query indices are then evaluated with the context-only
128D block and the full 416D pre-refinement key for all three deployed seeds.
No distance threshold is transferred between descriptor spaces.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DUMP_DIRS = {
    "seed1": REPO / "results/_final416_seed1",
    "seed2": REPO / "results/_final416_seed2",
    "seed3": REPO / "results/_final416_seed3",
}
SEQUENCES = [
    "kitti_00",
    "kitti_05",
    "kitti_08",
    "nclt_2012-01-08",
    "nclt_2013-01-10",
    "helipr_Town01",
    "mulran_DCC03",
    "mulran_KAIST03",
    "mulran_Riverside03",
]
EXPECTED_QUERIES = {
    "kitti_00": 632,
    "kitti_05": 377,
    "kitti_08": 235,
    "nclt_2012-01-08": 1834,
    "nclt_2013-01-10": 181,
    "helipr_Town01": 1586,
    "mulran_DCC03": 2344,
    "mulran_KAIST03": 2909,
    "mulran_Riverside03": 2796,
}
DTH, SKIP, EPS, BATCH = 5.0, 30, 0.1, 256


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def headline_queries(poses: np.ndarray) -> np.ndarray:
    """Queries with at least one earlier positive outside the temporal window."""
    positions = poses[:, :3, 3]
    queries = []
    for q in range(SKIP, len(positions)):
        prior = positions[: q - SKIP + 1]
        if np.any(np.linalg.norm(prior - positions[q], axis=1) < DTH):
            queries.append(q)
    return np.asarray(queries, dtype=np.int64)


def candidate_masks(
    positions: np.ndarray, query: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(positions))
    valid = np.abs(indices - query) >= SKIP
    geo = np.linalg.norm(positions - positions[query], axis=1)
    positive = valid & (geo < DTH)
    negative = valid & (geo >= DTH)
    return valid, positive, negative


def fixed_spread_subset(
    raw: np.ndarray, poses: np.ndarray, queries: np.ndarray
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Select raw misses whose best true match lies beyond EPS in raw space."""
    positions = poses[:, :3, 3]
    fixed = []
    raw_misses = 0
    for start in range(0, len(queries), BATCH):
        q_batch = queries[start : start + BATCH]
        similarities = raw[q_batch] @ raw.T
        for row, query in zip(similarities, q_batch):
            _, positive, negative = candidate_masks(positions, int(query))
            best_positive = float(np.max(row[positive]))
            best_negative = float(np.max(row[negative]))
            best_positive_distance = np.sqrt(max(0.0, 2.0 - 2.0 * best_positive))
            is_miss = best_positive <= best_negative
            raw_misses += int(is_miss)
            if is_miss and best_positive_distance > EPS:
                fixed.append(int(query))
    fixed_array = np.asarray(fixed, dtype=np.int64)
    counts = {
        "n_queries": int(len(queries)),
        "n_raw_misses": int(raw_misses),
        "n_fixed_spread_misses": int(len(fixed_array)),
        "fixed_%_of_queries": round(100.0 * len(fixed_array) / len(queries), 4),
        "fixed_%_of_raw_misses": round(
            100.0 * len(fixed_array) / max(raw_misses, 1), 4
        ),
    }
    return fixed_array, counts


def evaluate(
    descriptor: np.ndarray, poses: np.ndarray, queries: np.ndarray
) -> dict[str, np.ndarray]:
    """Return per-query recovery, rank, and cosine-margin arrays."""
    positions = poses[:, :3, 3]
    success, rank, margin = [], [], []
    for start in range(0, len(queries), BATCH):
        q_batch = queries[start : start + BATCH]
        similarities = descriptor[q_batch] @ descriptor.T
        for row, query in zip(similarities, q_batch):
            valid, positive, negative = candidate_masks(positions, int(query))
            best_positive = float(np.max(row[positive]))
            best_negative = float(np.max(row[negative]))
            success.append(best_positive > best_negative)
            rank.append(1 + int(np.sum(row[valid] > best_positive)))
            margin.append(best_positive - best_negative)
    return {
        "success": np.asarray(success, dtype=bool),
        "rank": np.asarray(rank, dtype=np.int64),
        "margin": np.asarray(margin, dtype=np.float64),
    }


def summarize(arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    n = len(arrays["success"])
    if n == 0:
        return {"n": 0}
    rank = arrays["rank"]
    return {
        "n": n,
        "recovered_R@1": float(np.mean(arrays["success"])),
        "recovered_R@5": float(np.mean(rank <= 5)),
        "median_true_match_rank": float(np.median(rank)),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank)),
        "median_cosine_margin": float(np.median(arrays["margin"])),
    }


def seed_summary(
    summaries: dict[str, dict[str, float | int]]
) -> dict[str, float | int]:
    result: dict[str, float | int] = {"n": int(next(iter(summaries.values()))["n"])}
    for metric in (
        "recovered_R@1",
        "recovered_R@5",
        "median_true_match_rank",
        "mean_reciprocal_rank",
        "median_cosine_margin",
    ):
        values = np.asarray([float(v[metric]) for v in summaries.values()])
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values, ddof=1))
    return result


def concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([part[key] for part in parts])
        for key in ("success", "rank", "margin")
    }


def main() -> None:
    report: dict[str, object] = {
        "protocol": {
            "subset": (
                "288D raw cosine miss with best true-match L2 distance > 0.1"
            ),
            "query_set": "headline queries with an earlier <5 m positive",
            "gallery": "all candidates with |i-j| >= 30",
            "evaluation": (
                "fixed query indices; no cross-space distance-threshold comparison"
            ),
            "seeds": list(DUMP_DIRS),
        },
        "sequences": {},
    }
    pooled: dict[str, dict[str, list[dict[str, np.ndarray]]]] = {
        group: {
            seed: {"context128": [], "full416": []}
            for seed in DUMP_DIRS
        }
        for group in ("all", "sparse")
    }
    pooled_counts = {
        group: {
            "n_queries": 0,
            "n_raw_misses": 0,
            "n_fixed_spread_misses": 0,
        }
        for group in ("all", "sparse")
    }

    for sequence in SEQUENCES:
        loaded = {
            seed: np.load(directory / f"{sequence}.npz")
            for seed, directory in DUMP_DIRS.items()
        }
        poses = loaded["seed1"]["poses"]
        embeddings = {seed: data["emb416"] for seed, data in loaded.items()}
        for seed, data in loaded.items():
            if not np.array_equal(poses, data["poses"]):
                raise ValueError(f"pose mismatch: {sequence}/{seed}")
            if not np.array_equal(
                embeddings["seed1"][:, :288], data["emb416"][:, :288]
            ):
                raise ValueError(f"invariant-key mismatch: {sequence}/{seed}")

        queries = headline_queries(poses)
        if len(queries) != EXPECTED_QUERIES[sequence]:
            raise ValueError(
                f"query-count mismatch for {sequence}: "
                f"{len(queries)} != {EXPECTED_QUERIES[sequence]}"
            )
        raw = normalize(embeddings["seed1"][:, :288])
        fixed, counts = fixed_spread_subset(raw, poses, queries)
        sequence_report: dict[str, object] = {
            "counts": counts,
            "context128": {},
            "full416": {},
        }
        groups = ["all"]
        if sequence.startswith(("nclt_", "helipr_")):
            groups.append("sparse")
        for group in groups:
            for key in pooled_counts[group]:
                pooled_counts[group][key] += int(counts[key])

        for seed, embedding in embeddings.items():
            evaluated = {
                "context128": evaluate(normalize(embedding[:, 288:]), poses, fixed),
                "full416": evaluate(normalize(embedding), poses, fixed),
            }
            for representation, arrays in evaluated.items():
                sequence_report[representation][seed] = summarize(arrays)
                for group in groups:
                    pooled[group][seed][representation].append(arrays)

        for representation in ("context128", "full416"):
            sequence_report[representation]["3seed"] = seed_summary(
                {
                    seed: sequence_report[representation][seed]
                    for seed in DUMP_DIRS
                }
            )
        report["sequences"][sequence] = sequence_report
        context = sequence_report["context128"]["3seed"]
        full = sequence_report["full416"]["3seed"]
        print(
            f"{sequence:<23} fixed={counts['n_fixed_spread_misses']:4d}/"
            f"{counts['n_queries']:4d}  "
            f"ctx R@1={context['recovered_R@1_mean']:.3f}"
            f"+/-{context['recovered_R@1_std']:.3f}  "
            f"full R@1={full['recovered_R@1_mean']:.3f}"
            f"+/-{full['recovered_R@1_std']:.3f}",
            flush=True,
        )

    aggregate = {}
    for group in ("all", "sparse"):
        group_report: dict[str, object] = {"counts": pooled_counts[group]}
        for representation in ("context128", "full416"):
            per_seed = {
                seed: summarize(concatenate(pooled[group][seed][representation]))
                for seed in DUMP_DIRS
            }
            group_report[representation] = {
                **per_seed,
                "3seed": seed_summary(per_seed),
            }
        aggregate[group] = group_report
    report["aggregate"] = aggregate

    output = REPO / "results/_spread_recovery_fixed_subset.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    for group in ("all", "sparse"):
        counts = aggregate[group]["counts"]
        context = aggregate[group]["context128"]["3seed"]
        full = aggregate[group]["full416"]["3seed"]
        print(
            f"[{group}] fixed={counts['n_fixed_spread_misses']}/"
            f"{counts['n_queries']}  "
            f"context R@1={context['recovered_R@1_mean']:.4f}"
            f"+/-{context['recovered_R@1_std']:.4f}  "
            f"full R@1={full['recovered_R@1_mean']:.4f}"
            f"+/-{full['recovered_R@1_std']:.4f}",
            flush=True,
        )
    print(f"WROTE {output.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
