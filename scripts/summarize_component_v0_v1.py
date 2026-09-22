#!/usr/bin/env python3
"""Summarize three-seed v0-v1 component results without refitting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_variant(path: Path, variant: str) -> dict:
    payload = json.loads(path.read_text())
    if variant not in payload:
        raise ValueError(f"{path}: missing {variant}")
    return payload[variant]


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "values": [float(value) for value in array],
    }


def seed_summary(run: dict) -> dict:
    recalls = [row["recall@1"] for row in run["per_sequence"].values()]
    return {
        "query_weighted_recall_at_1": float(run["avg_recall@1_query_weighted"]),
        "sequence_macro_recall_at_1": float(np.mean(recalls)),
        "worst_sequence_recall_at_1": float(np.min(recalls)),
        "cross_sequence_sample_std": float(np.std(recalls, ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed1", type=Path, default=Path("results/deployed_ablation.json"))
    parser.add_argument("--supplementary-dir", type=Path,
                        default=Path("results/supplementary"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/supplementary/component_v0_v1_3seed_summary.json"))
    args = parser.parse_args()

    runs: dict[str, list[dict]] = {"v0": [], "v1": []}
    sources: dict[str, list[str]] = {"v0": [], "v1": []}
    for variant in runs:
        seed_paths = [args.seed1] + [
            args.supplementary_dir / f"component_{variant}_seed{seed}.json"
            for seed in (2, 3)
        ]
        for path in seed_paths:
            runs[variant].append(load_variant(path, variant))
            sources[variant].append(str(path))

    sequence_sets = [set(run["per_sequence"]) for variant_runs in runs.values()
                     for run in variant_runs]
    if any(keys != sequence_sets[0] for keys in sequence_sets[1:]):
        raise ValueError("v0/v1 runs do not share an identical sequence set")
    query_counts = [run["n_queries_total"] for variant_runs in runs.values()
                    for run in variant_runs]
    if len(set(query_counts)) != 1:
        raise ValueError("v0/v1 runs do not share an identical query count")

    per_seed = {variant: [seed_summary(run) for run in variant_runs]
                for variant, variant_runs in runs.items()}
    metrics = tuple(per_seed["v0"][0])
    aggregate = {
        variant: {metric: stats([row[metric] for row in rows])
                  for metric in metrics}
        for variant, rows in per_seed.items()
    }
    paired_delta = {
        metric: stats([
            per_seed["v1"][seed][metric] - per_seed["v0"][seed][metric]
            for seed in range(3)
        ])
        for metric in metrics
    }
    sequence_delta = {}
    for sequence in sorted(sequence_sets[0]):
        values = [
            runs["v1"][seed]["per_sequence"][sequence]["recall@1"]
            - runs["v0"][seed]["per_sequence"][sequence]["recall@1"]
            for seed in range(3)
        ]
        sequence_delta[sequence] = stats(values)

    report = {
        "protocol": {
            "n_queries_total": query_counts[0],
            "metric": "raw 416D cosine Recall@1",
            "temporal_edge_mode": "bidirectional",
            "scope": "six-sequence component-ablation protocol",
        },
        "source_artifacts": sources,
        "per_seed": per_seed,
        "three_seed_summary": aggregate,
        "paired_v1_minus_v0": paired_delta,
        "per_sequence_paired_v1_minus_v0": sequence_delta,
        "interpretation_limit": (
            "This descriptive summary reports observed seed variation; it does "
            "not establish a superiority claim or a statistical test."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for metric, values in paired_delta.items():
        print(f"v1-v0 {metric}: {values['mean']:+.4f} +/- {values['sample_std']:.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
