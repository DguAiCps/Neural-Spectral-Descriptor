#!/usr/bin/env python3
"""Prepare the fixed NSD-only environment-disjoint supplementary protocol.

The primary evaluation keeps sensor families represented during training while
holding out named environments:

* KITTI: train 01/02/06/07, evaluate 00/05/08.
* HeLiPR: train Bridge01/Roundabout01, evaluate Town01.
* MulRan/HeLiPR overlap: train Sejong01/02/03 only, evaluate
  DCC03/KAIST03/Riverside03.

NCLT has no second environment in the available split. Its two evaluation
sequences are retained as a separate place-plus-unseen-sensor diagnostic and
are excluded from the primary seven-sequence aggregate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from train_multi_dataset import (  # noqa: E402
    apply_encoder_preset,
    compute_cache_key,
)


TRAIN = {
    "kitti": ["01", "02", "06", "07"],
    "helipr": ["Bridge01", "Roundabout01"],
    "mulran": ["Sejong01", "Sejong02", "Sejong03"],
}
PRIMARY_EVAL = [
    "kitti/00",
    "kitti/05",
    "kitti/08",
    "helipr/Town01",
    "mulran/DCC03",
    "mulran/KAIST03",
    "mulran/Riverside03",
]
CONFOUNDED_EVAL = ["nclt/2012-01-08", "nclt/2013-01-10"]
REMOVED_OVERLAP = {
    "nclt": [
        "2012-05-11",
        "2012-08-04",
        "2012-11-04",
        "2012-11-16",
        "2013-02-23",
    ],
    "helipr": ["Town02", "DCC04", "KAIST04", "Riverside04"],
    "mulran": [
        "DCC01",
        "DCC02",
        "KAIST01",
        "KAIST02",
        "Riverside01",
        "Riverside02",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_path(
    cache_root: Path,
    cache_keys: dict[str, str],
    dataset: str,
    sequence: str,
    validation: bool,
) -> Path:
    role = "val_" if validation else ""
    return cache_root / (
        f"cache_{cache_keys[dataset]}_{dataset}_{role}{sequence}.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=REPO / "configs/_abl_deployed_v0.yaml",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO / "data/preprocessed",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=(
            REPO
            / "results/supplementary/place_disjoint_nsd/fixed_protocol.yaml"
        ),
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=(
            REPO
            / "results/supplementary/place_disjoint_nsd/split_manifest.json"
        ),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.base_config.read_text())
    config["data"]["cache_dir"] = str(args.cache_root.relative_to(REPO))
    train_blocks = []
    for dataset, sequences in TRAIN.items():
        original = next(
            block
            for block in config["data"]["datasets"]["train"]
            if block["type"] == dataset
        )
        block = dict(original)
        block["sequences"] = sequences
        train_blocks.append(block)
    config["data"]["datasets"]["train"] = train_blocks

    # The reported checkpoint is fixed before training: epoch 100 final_model.
    # Validation runs at epochs 1 and 100 for auditing but cannot stop training.
    config["gnn"]["patience"] = 1000
    config["training"]["n_epochs"] = 100
    config["training"]["validate_every_n_epochs"] = 100

    effective = apply_encoder_preset(config, "no_interdiff")
    cache_keys = {
        dataset: compute_cache_key(effective, dataset)
        for dataset in ("kitti", "nclt", "helipr", "mulran")
    }

    required = []
    for dataset, sequences in TRAIN.items():
        for sequence in sequences:
            required.append(
                cache_path(
                    args.cache_root,
                    cache_keys,
                    dataset,
                    sequence,
                    validation=False,
                )
            )
    for item in PRIMARY_EVAL + CONFOUNDED_EVAL:
        dataset, sequence = item.split("/", 1)
        required.append(
            cache_path(
                args.cache_root,
                cache_keys,
                dataset,
                sequence,
                validation=True,
            )
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required cache files are missing:\n" + "\n".join(missing)
        )

    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(yaml.safe_dump(config, sort_keys=False))

    manifest = {
        "protocol": "environment-disjoint NSD-only control",
        "status": "pre-registered before training",
        "checkpoint_rule": (
            "fixed epoch-100 final_model.pth; best_model.pth is not reportable"
        ),
        "training": TRAIN,
        "primary_evaluation": PRIMARY_EVAL,
        "primary_interpretation": (
            "place-disjoint across named environments with the evaluation "
            "sensor family represented during training"
        ),
        "confounded_evaluation": CONFOUNDED_EVAL,
        "confounded_interpretation": (
            "NCLT place holdout also removes HDL-32E training; report "
            "separately as place-plus-unseen-sensor transfer"
        ),
        "removed_environment_overlap": REMOVED_OVERLAP,
        "known_limitations": [
            "Training-data volume changes with environment removal.",
            "No learned baseline is retrained on this split.",
            "KITTI route disjointness follows the repository's declared split.",
            "Results support a supplementary robustness control, not a "
            "Table-1 replacement.",
        ],
        "base_config": str(args.base_config.relative_to(REPO)),
        "base_config_sha256": sha256(args.base_config),
        "generated_config": str(args.config_out.relative_to(REPO)),
        "generated_config_sha256": sha256(args.config_out),
        "encoder_preset": "no_interdiff",
        "cache_root": str(args.cache_root.relative_to(REPO)),
        "cache_keys": cache_keys,
        "required_cache_files": [
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256(path),
            }
            for path in required
        ],
        "preparation_command": shlex.join(sys.argv),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.config_out}")
    print(f"Wrote {args.manifest_out}")


if __name__ == "__main__":
    main()
