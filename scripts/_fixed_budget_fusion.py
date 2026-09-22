#!/usr/bin/env python3
"""Evaluate candidate-source ablations under one fixed total candidate budget.

This script separates candidate recall from optional fixed-score reranking. It
does not sweep or select fusion weights on held-out sequences. A full reported
run must explicitly request all nine headline tags; smaller tag sets are smoke
tests only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from _height_candidate_fusion import (  # noqa: E402
    DTH,
    JOBS,
    SKIP,
    align_bank,
    align_scores,
    bin288,
    column_align_bank,
    column_align_scores,
    height_cache,
    l2,
    make_model,
    minmax,
    nsd_embedding,
    octave_key,
)
from run_kitti_operating_point import _find_queries, _make_encoder, _topk_cosine  # noqa: E402

HEADLINE_TAGS = ("K00", "K05", "K08", "N12", "N13", "TOWN", "DCC", "KAI", "RIV")
MODES = ("nsd_index", "height_index", "split_union")


def parse_cache_overrides(values: list[str]) -> dict[str, Path]:
    out = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected TAG=PATH cache override, got {value!r}")
        tag, path = value.split("=", 1)
        if tag not in JOBS:
            raise ValueError(f"unknown cache tag {tag!r}")
        out[tag] = Path(path)
    return out


def unique_take(primary: np.ndarray, secondary: np.ndarray, budget: int) -> np.ndarray:
    """Take exactly up to ``budget`` unique candidates in deterministic order."""
    chosen = []
    seen = set()
    for index in np.concatenate((primary, secondary)):
        value = int(index)
        if value not in seen:
            chosen.append(value)
            seen.add(value)
            if len(chosen) == budget:
                break
    return np.asarray(chosen, dtype=np.int64)


def candidates(nsd: np.ndarray, height: np.ndarray, query: int,
               budget: int, mode: str) -> np.ndarray:
    ranked_nsd = _topk_cosine(nsd, query, budget + 2 * SKIP, SKIP)[:budget]
    ranked_height = _topk_cosine(height, query, budget + 2 * SKIP, SKIP)[:budget]
    if mode == "nsd_index":
        return ranked_nsd
    if mode == "height_index":
        return ranked_height
    left = budget // 2
    right = budget - left
    return unique_take(ranked_nsd[:left], ranked_height[:right], budget)


def fixed_score(nsd: np.ndarray, height: np.ndarray, hfft: np.ndarray,
                hnorm: np.ndarray, column_bank, query: int,
                candidates_: np.ndarray, weights: tuple[float, float, float, float]) -> np.ndarray:
    nsd_score = minmax(nsd[candidates_] @ nsd[query])
    height_score = minmax(height[candidates_] @ height[query])
    alignment_score = minmax(align_scores(hfft, hnorm, query, candidates_))
    column_score = minmax(column_align_scores(column_bank, query, candidates_))
    return (weights[0] * nsd_score + weights[1] * height_score
            + weights[2] * alignment_score + weights[3] * column_score)


def evaluate(poses: np.ndarray, nsd: np.ndarray, height: np.ndarray,
             hfft: np.ndarray, hnorm: np.ndarray, column_bank, budget: int,
             score_weights: tuple[float, float, float, float] | None) -> dict:
    positions = poses[:, :3, 3]
    queries = _find_queries(poses, DTH, SKIP)
    totals = {mode: {"candidate_oracle_hits": 0, "fixed_score_hits": 0}
              for mode in MODES}
    for number, (query, _) in enumerate(queries):
        for mode in MODES:
            pool = candidates(nsd, height, query, budget, mode)
            good = np.linalg.norm(positions[pool] - positions[query], axis=1) < DTH
            totals[mode]["candidate_oracle_hits"] += int(good.any())
            if score_weights is not None and len(pool):
                ranked = int(np.argmax(fixed_score(
                    nsd, height, hfft, hnorm, column_bank, query, pool, score_weights)))
                totals[mode]["fixed_score_hits"] += int(good[ranked])
        if number and number % 500 == 0:
            print(f"  queries {number}/{len(queries)}", flush=True)
    n_queries = max(len(queries), 1)
    return {
        "n_queries": len(queries),
        "modes": {
            mode: {
                "candidate_oracle_recall_at_1": values["candidate_oracle_hits"] / n_queries,
                "fixed_score_recall_at_1": (
                    values["fixed_score_hits"] / n_queries if score_weights is not None else None
                ),
            }
            for mode, values in totals.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", required=True, choices=sorted(JOBS))
    parser.add_argument("--candidate-budgets", nargs="+", type=int,
                        default=(50, 100, 200))
    parser.add_argument("--height-variant", choices=("legacy240", "raw_height240", "height240"),
                        default="raw_height240")
    parser.add_argument("--seed", choices=(1, 2, 3), type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1,
                        help="Torch CPU threads (keep low while GPU training mines triplets).")
    parser.add_argument("--fixed-score-weights", nargs=4, type=float, default=None,
                        metavar=("W_NSD", "W_HEIGHT", "W_ALIGN", "W_COLUMN"))
    parser.add_argument("--cache-override", action="append", default=[], metavar="TAG=PATH")
    parser.add_argument("--require-complete-nine", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if any(budget <= 0 for budget in args.candidate_budgets):
        parser.error("candidate budgets must be positive")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    if args.require_complete_nine and set(args.tags) != set(HEADLINE_TAGS):
        parser.error("--require-complete-nine requires exactly the nine headline tags")
    overrides = parse_cache_overrides(args.cache_override)
    score_weights = (tuple(args.fixed_score_weights)
                     if args.fixed_score_weights is not None else None)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    cfg, model, classifier, classifier_blob = make_model(args.device, args.seed)
    encoder = _make_encoder(cfg, args.device)
    report = {
        "protocol": {
            "candidate_budgets": args.candidate_budgets,
            "candidate_modes": list(MODES),
            "height_variant": args.height_variant,
            "seed": args.seed,
            "device": args.device,
            "threads": args.threads,
            "fixed_score_weights": score_weights,
            "fixed_total_budget": True,
            "scope": "complete nine-sequence" if args.require_complete_nine else "smoke/incomplete; do not report as headline",
        },
        "sequences": {},
    }
    for tag in args.tags:
        sensor, sequence, cached = JOBS[tag]
        cache_path = overrides.get(tag, REPO / cached)
        print(f"\n=== {tag} ({cache_path}) ===", flush=True)
        cache = np.load(cache_path)
        d288 = bin288(cache["fft_magnitudes"].astype(np.float32))
        auxiliary = height_cache(tag, sensor, sequence, cache, encoder)
        if args.height_variant == "legacy240":
            height = l2(auxiliary["legacy_key"].astype(np.float32))
        elif args.height_variant == "raw_height240":
            height = octave_key(auxiliary["raw_height"].astype(np.float32))
        else:
            height = octave_key(auxiliary["height"].astype(np.float32))
        nsd = nsd_embedding(cache, sensor, d288, auxiliary["nsd_layout"].astype(np.float32),
                            cfg, model, classifier, classifier_blob, args.device)
        hfft, hnorm = align_bank(auxiliary["height"].astype(np.float32)[:, None], center=True)
        column_bank = column_align_bank(auxiliary["raw_height"].astype(np.float32))
        report["sequences"][tag] = {
            "sensor": sensor,
            "sequence": sequence,
            "cache_path": str(cache_path),
            "cache_scan_ids": int(len(cache["scan_ids"])),
            "budgets": {
                str(budget): evaluate(cache["poses"], nsd, height, hfft, hnorm,
                                      column_bank, budget, score_weights)
                for budget in args.candidate_budgets
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
