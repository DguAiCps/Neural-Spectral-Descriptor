#!/usr/bin/env python3
"""Aggregate fixed-epoch NSD place-disjoint evaluations across three seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PRIMARY = [
    "kitti/00",
    "kitti/05",
    "kitti/08",
    "helipr/Town01",
    "mulran/DCC03",
    "mulran/KAIST03",
    "mulran/Riverside03",
]
CONFOUNDED = ["nclt/2012-01-08", "nclt/2013-01-10"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted(per_sequence: dict, names: list[str]) -> tuple[float, int]:
    count = sum(per_sequence[name]["n_queries"] for name in names)
    score = sum(
        per_sequence[name]["recall@1"]
        * per_sequence[name]["n_queries"]
        for name in names
    )
    return score / count, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO / "results/supplementary/place_disjoint_nsd",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            REPO
            / "results/supplementary/place_disjoint_nsd/summary_3seed.json"
        ),
    )
    args = parser.parse_args()

    rows = []
    for seed in (1, 2, 3):
        path = args.input_dir / f"eval_seed{seed}.json"
        payload = json.loads(path.read_text())
        row = payload["v0"]
        if row["provenance"]["training_seed"] != seed:
            raise ValueError(f"{path}: training seed mismatch")
        names = set(row["per_sequence"])
        expected = set(PRIMARY + CONFOUNDED)
        if names != expected:
            raise ValueError(
                f"{path}: sequence set mismatch: {sorted(names ^ expected)}"
            )
        primary, primary_n = weighted(row["per_sequence"], PRIMARY)
        nclt, nclt_n = weighted(row["per_sequence"], CONFOUNDED)
        rows.append(
            {
                "seed": seed,
                "primary_7seq_recall@1": primary,
                "primary_n_queries": primary_n,
                "nclt_confounded_recall@1": nclt,
                "nclt_n_queries": nclt_n,
                "per_sequence": row["per_sequence"],
                "evaluation_artifact": str(path.relative_to(REPO)),
                "evaluation_artifact_sha256": sha256(path),
            }
        )

    primary_values = np.asarray(
        [row["primary_7seq_recall@1"] for row in rows]
    )
    nclt_values = np.asarray(
        [row["nclt_confounded_recall@1"] for row in rows]
    )
    output = {
        "protocol": {
            "key": "raw 416D NSD key",
            "checkpoint": "fixed epoch-100 final_model.pth",
            "metric": "cosine Recall@1",
            "primary_sequences": PRIMARY,
            "nclt_confounded_sequences": CONFOUNDED,
            "interpretation": (
                "Primary aggregate is an NSD-only environment-disjoint "
                "robustness control. NCLT is separate because place and "
                "unseen-HDL-32 transfer are confounded."
            ),
        },
        "seeds": rows,
        "primary_7seq": {
            "mean_recall@1": float(primary_values.mean()),
            "sample_std": float(primary_values.std(ddof=1)),
            "values": primary_values.tolist(),
        },
        "nclt_confounded": {
            "mean_recall@1": float(nclt_values.mean()),
            "sample_std": float(nclt_values.std(ddof=1)),
            "values": nclt_values.tolist(),
        },
        "aggregation_command": shlex.join(sys.argv),
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
