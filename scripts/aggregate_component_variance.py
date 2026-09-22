"""Aggregate the three-seed deployed component-ablation evaluation artifacts.

Seed 1 is read from ``results/deployed_ablation.json``; seeds 2 and 3 are
the separately evaluated supplementary artifacts.  The script rejects mixed
protocols or query counts so reported variability cannot silently combine
incomparable runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = ("v0", "v1", "v2", "v3", "v4", "v5")


def canonical_protocol(run: dict) -> tuple[dict, bool]:
    """Return a comparable protocol and record an old-artifact omission.

    The frozen seed-1 ablation artifact predates the explicit
    ``temporal_edge_mode`` field.  Its graph edge counts match the later
    bidirectional evaluations, which is also the evaluator's default.  Keep
    that compatibility inference local to the aggregate rather than silently
    weakening the protocol gate.
    """
    protocol = dict(run["protocol"])
    inferred = "temporal_edge_mode" not in protocol
    if inferred:
        protocol["temporal_edge_mode"] = "bidirectional"
    return protocol, inferred


def read_single_variant(path: Path, variant: str) -> dict:
    payload = json.loads(path.read_text())
    if variant not in payload:
        raise ValueError(f"{path}: missing variant {variant}")
    return payload[variant]


def sample_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "sample_std": float(arr.std(ddof=1)),
        "n_seeds": int(arr.size),
        "values": [float(x) for x in arr],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed1", type=Path,
                        default=Path("results/deployed_ablation.json"))
    parser.add_argument("--supplementary-dir", type=Path,
                        default=Path("results/supplementary"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/supplementary/component_variance_3seed.json"))
    args = parser.parse_args()

    inputs: dict[str, dict[int, tuple[Path, dict]]] = {v: {} for v in VARIANTS}
    for variant in VARIANTS:
        inputs[variant][1] = (args.seed1, read_single_variant(args.seed1, variant))
        for seed in (2, 3):
            path = args.supplementary_dir / f"component_{variant}_seed{seed}.json"
            inputs[variant][seed] = (path, read_single_variant(path, variant))

    reference = inputs["v0"][1][1]
    reference_protocol, reference_protocol_inferred = canonical_protocol(reference)
    reference_queries = reference["n_queries_total"]
    reference_sequences = set(reference["per_sequence"])
    for variant, runs in inputs.items():
        for seed, (path, run) in runs.items():
            protocol, protocol_inferred = canonical_protocol(run)
            if protocol != reference_protocol:
                raise ValueError(f"{path}: protocol differs from v0 seed 1")
            if run["n_queries_total"] != reference_queries:
                raise ValueError(f"{path}: query count differs from v0 seed 1")
            if set(run["per_sequence"]) != reference_sequences:
                raise ValueError(f"{path}: sequence set differs from v0 seed 1")

    out: dict = {
        "protocol": reference_protocol,
        "protocol_normalization": {
            "temporal_edge_mode": "bidirectional",
            "seed1_inferred_from": (
                "legacy artifact omitted temporal_edge_mode; evaluator default "
                "and matching temporal-edge counts establish bidirectional mode"
                if reference_protocol_inferred else None
            ),
        },
        "n_queries_total": reference_queries,
        "seeds": [1, 2, 3],
        "source_artifacts": {},
        "variants": {},
        "paired_delta_vs_v0": {},
    }
    means: dict[str, list[float]] = {}
    for variant, runs in inputs.items():
        ordered = [runs[seed][1] for seed in (1, 2, 3)]
        values = [run["avg_recall@1_query_weighted"] for run in ordered]
        means[variant] = values
        per_sequence = {}
        for sequence in sorted(reference_sequences):
            per_sequence[sequence] = sample_stats(
                [run["per_sequence"][sequence]["recall@1"] for run in ordered]
            )
        out["source_artifacts"][variant] = {
            str(seed): {
                "path": str(runs[seed][0]),
                "training_seed": seed,
                "protocol_inferred": canonical_protocol(runs[seed][1])[1],
            }
            for seed in (1, 2, 3)
        }
        out["variants"][variant] = {
            "label": ordered[0]["label"],
            "query_weighted_recall_at_1": sample_stats(values),
            "per_sequence": per_sequence,
        }

    control = np.asarray(means["v0"], dtype=float)
    for variant in VARIANTS:
        if variant == "v0":
            continue
        deltas = np.asarray(means[variant], dtype=float) - control
        out["paired_delta_vs_v0"][variant] = sample_stats(deltas.tolist())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[ok] wrote {args.output}")
    for variant in VARIANTS:
        stats = out["variants"][variant]["query_weighted_recall_at_1"]
        print(f"  {variant}: {stats['mean']:.4f} +/- {stats['sample_std']:.4f}")


if __name__ == "__main__":
    main()
