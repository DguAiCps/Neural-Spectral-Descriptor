#!/usr/bin/env python3
"""Create a compact, hash-addressed ledger for completed supplementary work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/supplementary/evidence_package_20260731.json"
CHECKPOINT = "checkpoints/800d_4sensor_20260511_161726/best_model.pth"
CONFIG = "configs/training_multi_dataset.yaml (encoder preset no_interdiff)"
CROSS_SENSOR_CACHE = "data/preprocessed_cross_sensor_operating"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: str, **metadata: object) -> dict[str, object]:
    local = REPO / path
    if not local.exists():
        raise FileNotFoundError(local)
    return {"path": path, "sha256": sha256(local), **metadata}


def read(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def cache_hashes(*paths: str) -> dict[str, str]:
    hashes = {}
    for path in paths:
        local = REPO / path
        if not local.is_file():
            raise FileNotFoundError(local)
        hashes[path] = sha256(local)
    return hashes


def main() -> None:
    b1 = read("results/supplementary/fixed_budget_9seq_summary.json")
    component = read("results/supplementary/component_variance_3seed.json")
    component_summary = read("results/supplementary/component_v0_v1_3seed_summary.json")
    causal = [
        read(f"results/supplementary/causal_retrain_seed{seed}.json")["baseline"]
        for seed in (1, 2, 3)
    ]
    frozen = read("results/supplementary/frozen_past_to_current_seed1.json")["baseline"]
    ring_200 = read("results/supplementary/ringsharp_all_k200_t100.json")
    ring_1000 = read("results/supplementary/ringsharp_all_k1000_t100.json")
    lineage = read("results/supplementary/headline_data_lineage.json")

    causal_values = [row["avg_recall@1_query_weighted"] for row in causal]
    causal_mean = sum(causal_values) / len(causal_values)
    causal_sample_std = (
        sum((value - causal_mean) ** 2 for value in causal_values) / (len(causal_values) - 1)
    ) ** 0.5
    raw416_cache_hashes = cache_hashes(
        "data/preprocessed_cross_sensor_operating/kitti_operating_00_layout60_stride1.npz",
        "data/preprocessed_cross_sensor_operating/kitti_operating_05_layout60_stride1.npz",
        "data/preprocessed_cross_sensor_operating/kitti_operating_08_layout60_stride1.npz",
        "data/preprocessed_cross_sensor_operating/nclt_operating_2012-01-08_layout60_stride1.npz",
        "data/preprocessed_cross_sensor_operating/nclt_operating_2013-01-10_layout60_stride1.npz",
        "data/preprocessed_cross_sensor_operating/helipr_operating_Town01_layout60_stride1.npz",
    )

    payload = {
        "scope": {
            "purpose": "Reproducible evidence package for completed supplementary analyses.",
            "paper_positioning": (
                "Compact NSD competes with, or is on par with, the strongest learned reference."
            ),
            "prohibited_interpretations": [
                "No paired statistical-superiority claim.",
                "No partial paired comparison.",
                "No Table-1 superiority claim from regenerated-cache results.",
                "No end-to-end fusion claim from candidate-set oracle results.",
            ],
        },
        "seed_policy": [
            {
                "analysis": "B1 fixed-budget candidate-source ablation",
                "seeds": [1],
                "query_count": 12894,
                "aggregation": "single seed; nine-sequence query-weighted and per-sequence candidate-set oracle",
            },
            {
                "analysis": "component ablations",
                "seeds": [1, 2, 3],
                "query_count": 4845,
                "aggregation": "matched three-seed mean and sample standard deviation",
            },
            {
                "analysis": "causal retraining",
                "seeds": [1, 2, 3],
                "query_count": 4845,
                "aggregation": "three-seed mean and sample standard deviation",
            },
            {
                "analysis": "frozen causal control",
                "seeds": [1],
                "query_count": 4845,
                "aggregation": "single matched frozen checkpoint",
            },
            {
                "analysis": "RING#-L protocol sensitivity",
                "seeds": [1],
                "query_count": 12894,
                "aggregation": "single same-split checkpoint; not a multi-seed result",
            },
            {
                "analysis": "learned-baseline latency",
                "seeds": [],
                "query_count": 100,
                "aggregation": "100 timed queries per benchmark map after 10 warm-up queries",
            },
        ],
        "results": {
            "b1_fixed_budget": {
                "interpretation": b1["analysis"]["interpretation"],
                "fixed_score_status": b1["analysis"]["fixed_score_status"],
                "query_weighted_oracle": {
                    budget: row["query_weighted_candidate_oracle_recall_at_1"]
                    for budget, row in b1["results"].items()
                },
                "per_sequence_in": "results/supplementary/fixed_budget_9seq_summary.json",
            },
            "component_v0_v1": component_summary["paired_v1_minus_v0"],
            "causal": {
                "causal_retrain_query_weighted_recall_at_1": {
                    "values": causal_values,
                    "mean": causal_mean,
                    "sample_std": causal_sample_std,
                },
                "frozen_past_to_current_seed1": frozen["avg_recall@1_query_weighted"],
                "interpretation": (
                    "Retraining improves the frozen causal graph under the six-sequence "
                    "raw-416D protocol; it is reported descriptively."
                ),
            },
            "ringsharp_l": {
                "k200_query_weighted_recall_at_1": ring_200["_query_weighted_mean"]["recall@1"],
                "k1000_query_weighted_recall_at_1": ring_1000["_query_weighted_mean"]["recall@1"],
                "protocol": ring_200["_meta"]["protocol"],
            },
            "lineage": {
                "mulran_query_counts": {
                    tag: row["query_count"] for tag, row in lineage["mulran"].items()
                },
                "nclt_historical_canonical_cache_available": all(
                    row["canonical_exists"] for row in lineage["nclt"].values()
                ),
            },
        },
        "artifacts": [
            record(
                "results/supplementary/fixed_budget_9seq_summary.json",
                command=b1["provenance"]["execution_command"],
                config=b1["provenance"]["config"],
                checkpoint=b1["provenance"]["checkpoint"],
                cache_hashes=b1["provenance"]["cache_hashes"],
                seed=b1["provenance"]["seed"],
                query_count=12894,
            ),
            record(
                "results/supplementary/component_variance_3seed.json",
                command="bash scripts/resume_component_variance_queue.sh; python3 scripts/aggregate_component_variance.py",
                config=CONFIG,
                checkpoint="component_v{0..5}_seed{1..3}/best_model.pth; source paths recorded in artifact",
                cache_path=CROSS_SENSOR_CACHE,
                cache_hashes=raw416_cache_hashes,
                seed=component["seeds"],
                query_count=component["n_queries_total"],
            ),
            record(
                "results/deployed_ablation.json",
                command="bash scripts/_run_deployed_ablation.sh",
                config="per-variant config paths recorded in artifact",
                checkpoint="per-variant checkpoint paths recorded in artifact",
                cache_path=CROSS_SENSOR_CACHE,
                cache_hashes=raw416_cache_hashes,
                seed=[1],
                query_count=4845,
                note=(
                    "Legacy seed-1 source for all component variants; its temporal-edge "
                    "mode is inferred as bidirectional in component_variance_3seed.json."
                ),
            ),
            record(
                "results/supplementary/component_v0_v1_3seed_summary.json",
                command="python3 scripts/summarize_component_v0_v1.py",
                config=CONFIG,
                checkpoint="component v0/v1 source checkpoints recorded by component variance artifact",
                cache_path=CROSS_SENSOR_CACHE,
                cache_hashes=raw416_cache_hashes,
                seed=[1, 2, 3],
                query_count=4845,
            ),
            *[
                record(
                    f"results/supplementary/causal_retrain_seed{seed}.json",
                    command=row["provenance"]["evaluation_command"],
                    config=row["config"],
                    checkpoint=row["checkpoint"],
                    cache_path=row["provenance"]["cache_root"],
                    cache_hashes=raw416_cache_hashes,
                    seed=seed,
                    query_count=row["n_queries_total"],
                )
                for seed, row in zip((1, 2, 3), causal)
            ],
            record(
                "results/supplementary/frozen_past_to_current_seed1.json",
                command=frozen["provenance"]["evaluation_command"],
                config=frozen["config"],
                checkpoint=frozen["checkpoint"],
                cache_path=frozen["provenance"]["cache_root"],
                cache_hashes=raw416_cache_hashes,
                seed=1,
                query_count=frozen["n_queries_total"],
            ),
            record(
                "results/supplementary/ringsharp_all_k200_t100.json",
                command="python3 scripts/_ringsharp_eval.py --n-coarse 200 --n-trans 100 --all-nine",
                config="external RINGSharp-L same-split checkpoint; checkpoint path in artifact _meta",
                checkpoint=ring_200["_meta"]["weight"],
                cache_path=CROSS_SENSOR_CACHE,
                seed=1,
                query_count=ring_200["_query_weighted_mean"]["n_queries"],
            ),
            record(
                "results/supplementary/ringsharp_all_k1000_t100.json",
                command="python3 scripts/_ringsharp_eval.py --n-coarse 1000 --n-trans 100 --all-nine",
                config="external RINGSharp-L same-split checkpoint; checkpoint path in artifact _meta",
                checkpoint=ring_1000["_meta"]["weight"],
                cache_path=CROSS_SENSOR_CACHE,
                seed=1,
                query_count=ring_1000["_query_weighted_mean"]["n_queries"],
            ),
            record(
                "results/baseline_latency_bench.json",
                command="bash scripts/_bench_baseline_latency.sh",
                config="benchmark harness settings in artifact env",
                checkpoint="method-specific baseline checkpoints/configurations in artifact",
                cache_hashes=read("results/baseline_latency_bench.json")["env"]["cache_keys"],
                seed="not applicable",
                query_count=100,
            ),
            record(
                "results/supplementary/bispectrum_provenance.json",
                command="python3 scripts/audit_bispectrum_provenance.py",
                config="configs/training_kitti_phase_bispectrum_dualstream_fast.yaml",
                checkpoint="none found",
                cache_path="not applicable",
                seed="not applicable",
                query_count="not independently established",
            ),
            record(
                "results/supplementary/headline_data_lineage.json",
                command="python3 scripts/audit_headline_data_lineage.py",
                config=CONFIG,
                checkpoint=CHECKPOINT,
                cache_path=CROSS_SENSOR_CACHE,
                seed="not applicable",
                query_count={"DCC03": 2344, "KAIST03": 2909, "Riverside03": 2796},
            ),
            record(
                "docs/SUPPLEMENTARY_PAIRED_ANALYSIS_PROTOCOL_20260730.md",
                command="scripts/_headline_perquery_dump.py; scripts/_paired_headline_stats.py",
                config=CONFIG,
                checkpoint=CHECKPOINT,
                cache_path="candidate NCLT cache plus cross-sensor operating cache",
                seed=[1],
                query_count=12894,
                status="excluded: Town01 NSD and Riverside03 NSD-H exceed the one-hit Table-1 gate",
            ),
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
