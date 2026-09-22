#!/usr/bin/env python3
"""Collect the supplementary competitive-positioning experiments into a ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/supplementary/competitive_supplementary_evidence_20260731.json"
BASE_CONFIG = REPO / "configs/training_multi_dataset.yaml"
BASE_CHECKPOINT = REPO / "checkpoints/800d_4sensor_20260511_161726/best_model.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def record(path: str, **metadata: object) -> dict[str, object]:
    local = REPO / path
    return {"path": path, "sha256": sha256(local), **metadata}


def main() -> None:
    raw = read("results/supplementary/raw_grid_height_seed1.json")
    fusion = read("results/supplementary/fusion_tuple_sensitivity.json")
    pose = read("results/supplementary/pose_noise_sensitivity.json")
    mlp = read("results/supplementary/mean_pool_mlp_3seed.json")
    ident = read("results/supplementary/identifiability_3seed.json")
    raw_query_count = sum(raw[tag]["variants"]["raw_height240"]["n_queries"]
                          for tag in fusion["protocol"]["validation_tags"])
    payload = {
        "scope": {
            "positioning": "Compact NSD competes with, or is on par with, the strongest learned reference.",
            "excluded_claims": [
                "No paired statistical-superiority claim.",
                "No Table-1-exact or partial paired comparison.",
                "No historical Table-1 reproduction claim for regenerated NCLT results.",
            ],
        },
        "seed_policy": [
            {"analysis": "raw-grid height / fixed tuple", "seeds": [1], "query_count": raw_query_count,
             "aggregation": "single-seed regenerated-cache protocol; tuple chosen only on training tags"},
            {"analysis": "pose-noise sensitivity", "seeds": [0, 1, 2], "query_count": 4845,
             "aggregation": "three independent noise realizations per non-clean condition"},
            {"analysis": "mean-pool + MLP", "seeds": [1, 2, 3], "query_count": 4845,
             "aggregation": "three training seeds; final fixed epoch"},
            {"analysis": "identifiability", "seeds": [1, 2, 3], "query_count": "not applicable",
             "aggregation": "fixed-epsilon collision test; no qualifying pairs in current dump lineage"},
        ],
        "results": {
            "raw_grid_height": {
                "n_validation_queries": raw_query_count,
                "nclt_protocol": "regenerated candidate cache for N12/N13 only",
                "selected_tuple": fusion["selected_training_tuple"],
                "selected_tuple_validation_per_sequence": fusion["selected_tuple_validation_per_sequence"],
                "sensitivity_range": fusion["validation_range_over_all_training_selected_candidates"],
            },
            "pose_noise": {
                name: row["query_weighted_recall_at_1"]
                for name, row in pose["conditions"].items()
            },
            "mean_pool_mlp": mlp["three_seed_summary"],
            "identifiability": ident["aggregate"],
        },
        "artifacts": [
            record(
                "results/supplementary/raw_grid_height_seed1.json",
                command=(
                    "docker exec ring_baseline ... python3 scripts/_height_candidate_fusion.py "
                    "--device cuda --pool-k 100 --variants legacy240 raw_height240 height240 "
                    "--tags K00 K05 K08 N12 N13 TOWN DCC KAI RIV K01_TR K02_TR K06_TR K07_TR N0511_TR N0804_TR "
                    "with the cache overrides recorded in _meta"
                ),
                config="configs/training_multi_dataset.yaml with encoder preset no_interdiff",
                checkpoint=str(BASE_CHECKPOINT.relative_to(REPO)),
                checkpoint_sha256=sha256(BASE_CHECKPOINT),
                cache_paths_and_hashes={
                    tag: {"path": raw[tag]["cache_path"], "sha256": raw[tag]["cache_sha256"]}
                    for tag in fusion["protocol"]["training_tags"] + fusion["protocol"]["validation_tags"]
                },
                seed=raw["_meta"]["seed"],
                query_count={tag: raw[tag]["variants"]["raw_height240"]["n_queries"]
                             for tag in fusion["protocol"]["training_tags"] + fusion["protocol"]["validation_tags"]},
            ),
            record(
                "results/supplementary/fusion_tuple_sensitivity.json",
                command="python3 scripts/_summarize_fusion_tuple_sensitivity.py --input results/supplementary/raw_grid_height_seed1.json --variant raw_height240",
                config="training tags and selection rule recorded in artifact",
                checkpoint=str(BASE_CHECKPOINT.relative_to(REPO)),
                checkpoint_sha256=sha256(BASE_CHECKPOINT),
                cache_paths_and_hashes="in referenced raw-grid artifact",
                seed=1,
                query_count=raw_query_count,
            ),
            record(
                "results/supplementary/pose_noise_sensitivity.json",
                command=pose["provenance"]["command"],
                config=pose["provenance"]["config"],
                config_sha256=pose["provenance"]["config_sha256"],
                checkpoint=pose["provenance"]["checkpoint"],
                checkpoint_sha256=pose["provenance"]["checkpoint_sha256"],
                cache_paths_and_hashes="per sequence and noise seed in artifact",
                seed=pose["protocol"]["noise_seeds"],
                query_count=4845,
            ),
            record(
                "results/supplementary/mean_pool_mlp_3seed.json",
                command=mlp["provenance"]["command"],
                config="configs/training_multi_dataset.yaml train split, cache keys listed per seed",
                config_sha256=sha256(BASE_CONFIG),
                checkpoint="per-seed final checkpoints and hashes in artifact",
                cache_paths_and_hashes="training and six diagnostic evaluation caches recorded per seed",
                seed=mlp["provenance"]["seeds"],
                query_count=4845,
            ),
            record(
                "results/supplementary/identifiability_3seed.json",
                command=ident["provenance"]["command"],
                config="not applicable; analysis of precomputed deployed-key dumps",
                checkpoint=ident["provenance"]["checkpoints"],
                cache_paths_and_hashes="source dump SHA-256 recorded per sequence and seed",
                seed=ident["protocol"]["seeds"],
                query_count="not applicable; pair sampling over keyframes",
            ),
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
