#!/usr/bin/env python3
"""Aggregate the complete fixed-total-candidate-budget B1 result.

The output deliberately reports candidate-set oracle recall only unless the
input itself contains a fixed, train-calibrated reranking score.  It never
selects a held-out score tuple.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO / "results/supplementary/fixed_budget_9seq_regenerated_candidate_seed1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results/supplementary/fixed_budget_9seq_summary.json",
    )
    args = parser.parse_args()

    source = json.loads(args.input.read_text())
    protocol = source["protocol"]
    modes = protocol["candidate_modes"]
    if protocol.get("fixed_score_weights") is not None:
        raise ValueError("This aggregator is restricted to the oracle-only B1 artifact")
    if set(modes) != {"nsd_index", "height_index", "split_union"}:
        raise ValueError(f"unexpected candidate modes: {modes}")

    budgets = {}
    for budget in protocol["candidate_budgets"]:
        total_queries = 0
        weighted = {mode: 0.0 for mode in modes}
        per_sequence = {}
        for tag, row in source["sequences"].items():
            result = row["budgets"][str(budget)]
            n_queries = int(result["n_queries"])
            total_queries += n_queries
            values = {
                mode: float(result["modes"][mode]["candidate_oracle_recall_at_1"])
                for mode in modes
            }
            for mode, value in values.items():
                weighted[mode] += n_queries * value
            per_sequence[tag] = {"n_queries": n_queries, "oracle_recall_at_1": values}
        aggregate = {mode: value / total_queries for mode, value in weighted.items()}
        strongest_single = max(aggregate["nsd_index"], aggregate["height_index"])
        budgets[str(budget)] = {
            "n_queries": total_queries,
            "query_weighted_candidate_oracle_recall_at_1": aggregate,
            "split_union_gain_vs_best_single_pp": 100 * (aggregate["split_union"] - strongest_single),
            "per_sequence": per_sequence,
        }

    cache_hashes = {}
    for tag, row in source["sequences"].items():
        recorded = Path(row["cache_path"])
        relative = str(recorded).replace("/workspace/Neural-Spectral-Codec/", "")
        local = REPO / relative
        if not local.is_file():
            raise FileNotFoundError(
                f"cannot hash B1 cache for {tag}: recorded={recorded}, local={local}"
            )
        cache_hashes[tag] = {
            "recorded_path": str(recorded),
            "local_path": relative,
            "sha256": sha256(local),
        }
    output = {
        "analysis": {
            "name": "B1 fixed-total-candidate-budget candidate-source ablation",
            "interpretation": (
                "Candidate-set oracle recall only. The result tests whether the two "
                "candidate indices cover complementary true matches at the same total "
                "candidate budget; it does not establish end-to-end reranking Recall@1."
            ),
            "fixed_score_status": {
                "evaluated": False,
                "reason": (
                    "No persisted train-calibrated score tuple exists for the raw_height240 "
                    "candidate-source control. Held-out tuple selection is prohibited."
                ),
            },
        },
        "provenance": {
            "source_artifact": str(args.input.relative_to(REPO)),
            "source_artifact_sha256": sha256(args.input),
            "execution_command": (
                "NSD_DATA_ROOT=/workspace/data PYTHONDONTWRITEBYTECODE=1 python3 "
                "scripts/_fixed_budget_fusion.py --tags K00 K05 K08 N12 N13 TOWN DCC KAI RIV "
                "--require-complete-nine --candidate-budgets 50 100 200 "
                "--height-variant raw_height240 --seed 1 --device cuda --threads 2 "
                "--cache-override N12=data/preprocessed_nclt_regenerated_candidate_20260730/"
                "nclt_operating_2012-01-08_layout60_stride1.npz "
                "--cache-override N13=data/preprocessed_nclt_regenerated_candidate_20260730/"
                "nclt_operating_2013-01-10_layout60_stride1.npz"
            ),
            "config": "configs/training_multi_dataset.yaml with encoder preset no_interdiff",
            "checkpoint": "checkpoints/800d_4sensor_20260511_161726/best_model.pth",
            "cache_hashes": cache_hashes,
            "seed": protocol["seed"],
        },
        "results": budgets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.output}")
    for budget, row in budgets.items():
        print(
            f"budget={budget}: union gain={row['split_union_gain_vs_best_single_pp']:+.3f}pp "
            f"over the best single index"
        )


if __name__ == "__main__":
    main()
