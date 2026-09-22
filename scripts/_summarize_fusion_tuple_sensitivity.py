#!/usr/bin/env python3
"""Select raw-grid fusion tuples on training tags and report held-out sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TRAIN_TAGS = ["K01_TR", "K02_TR", "K06_TR", "K07_TR", "N0511_TR", "N0804_TR"]
VALIDATION_TAGS = ["K00", "K05", "K08", "N12", "N13", "TOWN", "DCC", "KAI", "RIV"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted(rows: dict[str, dict], tuple_key: str) -> float:
    numerator = sum(row["n_queries"] * row[tuple_key] for row in rows.values())
    denominator = sum(row["n_queries"] for row in rows.values())
    return numerator / denominator


def collect(source: dict, tags: list[str], variant: str) -> dict[str, dict]:
    rows = {}
    for tag in tags:
        if tag not in source:
            raise ValueError(f"missing tag {tag}; rerun raw-grid control with all required tags")
        values = source[tag]["variants"].get(variant)
        if values is None:
            raise ValueError(f"missing variant {variant} for {tag}")
        rows[tag] = values
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=REPO / "results/supplementary/raw_grid_height_seed1.json")
    parser.add_argument("--variant", default="raw_height240")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/supplementary/fusion_tuple_sensitivity.json")
    args = parser.parse_args()
    input_path = args.input
    if not input_path.is_absolute():
        input_path = REPO / input_path
    source = json.loads(input_path.read_text())
    train = collect(source, TRAIN_TAGS, args.variant)
    validation = collect(source, VALIDATION_TAGS, args.variant)
    forbidden = {"nsd", "height", "oracle_union", "n_queries", "_perquery"}
    tuples = sorted(set.intersection(*[
        {key for key in row if key not in forbidden} for row in train.values()
    ]))
    if not tuples:
        raise ValueError("no shared fusion tuple sweep in input artifact")
    ranking = []
    for tuple_key in tuples:
        ranking.append({
            "tuple": tuple_key,
            "training_query_weighted_recall_at_1": weighted(train, tuple_key),
            "validation_query_weighted_recall_at_1": weighted(validation, tuple_key),
        })
    ranking.sort(key=lambda row: (-row["training_query_weighted_recall_at_1"], row["tuple"]))
    selected = ranking[0]
    selected_per_sequence = {
        tag: {
            "recall_at_1": validation[tag][selected["tuple"]],
            "n_queries": validation[tag]["n_queries"],
        }
        for tag in VALIDATION_TAGS
    }
    report = {
        "protocol": {
            "variant": args.variant,
            "tuple_selection": "maximize query-weighted Recall@1 across fixed training tags only",
            "training_tags": TRAIN_TAGS,
            "validation_tags": VALIDATION_TAGS,
            "interpretation": (
                "Fixed-tuple sensitivity only. The selection criterion does not use validation tags; "
                "the source raw-grid artifact remains regenerated-cache protocol for N12/N13."
            ),
        },
        "provenance": {
            "source_artifact": str(input_path.relative_to(REPO)),
            "source_sha256": sha256(input_path),
            "source_meta": source.get("_meta", {}),
        },
        "selected_training_tuple": selected,
        "selected_tuple_validation_per_sequence": selected_per_sequence,
        "top_training_tuples": ranking[:args.top_k],
        "validation_range_over_all_training_selected_candidates": {
            "min": min(row["validation_query_weighted_recall_at_1"] for row in ranking),
            "max": max(row["validation_query_weighted_recall_at_1"] for row in ranking),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}; selected {selected['tuple']}")


if __name__ == "__main__":
    main()
