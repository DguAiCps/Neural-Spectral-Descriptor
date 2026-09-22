"""Correct metadata for a completed causal-retraining evaluation artifact.

This is intentionally metadata-only: it validates the expected checkpoint and
does not alter any per-query, per-sequence, or aggregate metric values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args()

    path = args.path or (REPO / "results" / "supplementary" /
                         f"causal_retrain_seed{args.seed}.json")
    payload = json.loads(path.read_text())
    result = payload["baseline"]
    expected = f"supplementary_causal_seed{args.seed}/best_model.pth"
    if not result["checkpoint"].endswith(expected):
        raise ValueError(f"{path}: unexpected checkpoint {result['checkpoint']}")
    if result["protocol"].get("temporal_edge_mode") != "past_to_current":
        raise ValueError(f"{path}: not a causal past-to-current evaluation")

    result["label"] = f"causal retraining (past_to_current), seed {args.seed}"
    result["provenance"] = {
        "training_seed": args.seed,
        "evaluation_command": (
            "python3 scripts/_eval_deployed_ablation.py baseline "
            "--config configs/training_multi_dataset.yaml "
            f"--checkpoint checkpoints/supplementary_causal_seed{args.seed}/best_model.pth "
            f"--label 'causal retraining (past_to_current), seed {args.seed}' "
            f"--training-seed {args.seed} --temporal-edge-mode past_to_current "
            f"--out results/supplementary/causal_retrain_seed{args.seed}.json"
        ),
        "cache_root": "/workspace/Neural-Spectral-Codec/data/preprocessed_cross_sensor_operating",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[ok] metadata corrected: {path}")


if __name__ == "__main__":
    main()
