#!/usr/bin/env python3
"""Collect the sequential sensor-GAT + physics3 experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _first_metric(d: Any, names: tuple[str, ...]) -> float | None:
    if isinstance(d, dict):
        for name in names:
            if name in d and isinstance(d[name], (int, float)):
                return float(d[name])
        for value in d.values():
            found = _first_metric(value, names)
            if found is not None:
                return found
    elif isinstance(d, list):
        for value in d:
            found = _first_metric(value, names)
            if found is not None:
                return found
    return None


def _sequence_r1(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        if key.startswith("_"):
            continue
        r1 = _first_metric(value, ("R@1", "recall@1", "recall_at_1"))
        if r1 is not None:
            out[str(key)] = r1
    return out


def _best_reranker_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    metrics = payload.get("best_metrics", payload.get("metrics", payload))
    if not isinstance(metrics, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        r1 = _first_metric(value, ("recall@1", "R@1", "recall_at_1"))
        if r1 is not None:
            out[str(key)] = r1
    avg = _first_metric(metrics.get("_average", {}), ("recall@1", "R@1", "recall_at_1"))
    if avg is not None:
        out["_average"] = avg
    return out


def _best_phase_sketch_metrics(payload: Any) -> tuple[dict[str, float], dict[str, str]]:
    """Extract the best phase-sketch fusion R@1 per sequence.

    The KITTI phase-sketch evaluator stores baseline metrics first
    (``raw``/``final``) and the actual sketch rerank under
    ``raw_phase_sketch`` / ``final_phase_sketch``. A generic recursive search
    therefore reports the baseline by accident. This function intentionally
    selects the best ``phase_sketch_fusion_*`` row.
    """
    if not isinstance(payload, dict):
        return {}, {}

    metrics: dict[str, float] = {}
    keys: dict[str, str] = {}
    for seq, seq_payload in payload.items():
        if str(seq).startswith("_") or not isinstance(seq_payload, dict):
            continue
        best_r1: float | None = None
        best_key: str | None = None
        for block_name in ("final_phase_sketch", "raw_phase_sketch"):
            block = seq_payload.get(block_name)
            if not isinstance(block, dict):
                continue
            for key, value in block.items():
                if key.startswith("_") or not isinstance(value, dict):
                    continue
                r1 = _first_metric(value, ("R@1", "recall@1", "recall_at_1"))
                if r1 is None:
                    continue
                if best_r1 is None or r1 > best_r1:
                    best_r1 = r1
                    best_key = f"{block_name}.{key}"
        if best_r1 is not None:
            metrics[str(seq)] = best_r1
            if best_key is not None:
                keys[str(seq)] = best_key
    return metrics, keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--kitti-gat", required=True)
    parser.add_argument("--kitti-sketch", required=True)
    parser.add_argument("--kitti-reranker", required=True)
    parser.add_argument("--nclt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    kitti_gat = _load(args.kitti_gat)
    kitti_sketch = _load(args.kitti_sketch)
    kitti_reranker = _load(args.kitti_reranker)
    nclt = _load(args.nclt)
    nclt_metrics = nclt.get("metrics", nclt) if isinstance(nclt, dict) else nclt
    kitti_sketch_r1, kitti_sketch_best_keys = _best_phase_sketch_metrics(kitti_sketch)

    summary = {
        "run_id": args.run_id,
        "storage_dim": {
            "retrieval_key": 416,
            "phase_sketch": 384,
            "total": 800,
        },
        "files": {
            "kitti_gat": args.kitti_gat,
            "kitti_sketch": args.kitti_sketch,
            "kitti_reranker": args.kitti_reranker,
            "nclt": args.nclt,
        },
        "kitti_gat_only_r1": _sequence_r1(kitti_gat),
        "kitti_physics3_sketch_r1": kitti_sketch_r1,
        "kitti_physics3_sketch_best_keys": kitti_sketch_best_keys,
        "kitti_learned_reranker_r1": _best_reranker_metrics(kitti_reranker),
        "nclt_zero_shot_r1": _best_reranker_metrics(nclt_metrics),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
