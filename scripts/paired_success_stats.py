#!/usr/bin/env python3
"""Paired McNemar and moving-block-bootstrap tests for two retrieval dumps."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_records(path: Path) -> dict[int, bool]:
    data = json.loads(path.read_text())
    return {int(r["query_idx"]): bool(r["success_at_k1"]) for r in data["records"]}


def mcnemar_exact(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    k = min(left_only, right_only)
    log_p = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             - n * math.log(2.0) for i in range(n + 1)]
    return min(1.0, sum(math.exp(x) for i, x in enumerate(log_p)
                        if i <= k or i >= n - k))


def bootstrap(pairs: list[tuple[np.ndarray, np.ndarray]], block: int, reps: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    total = sum(len(left) for left, _ in pairs)
    out = np.empty(reps)
    for rep in range(reps):
        left_hits = right_hits = 0
        for left, right in pairs:
            n = len(left)
            starts = rng.integers(0, max(1, n - block + 1), size=math.ceil(n / block))
            idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
            left_hits += int(left[idx].sum())
            right_hits += int(right[idx].sum())
        out[rep] = (left_hits - right_hits) / total
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, required=True,
                        help="Directory containing <sequence>.json records")
    parser.add_argument("--right-dir", type=Path, required=True,
                        help="Directory containing identically named records")
    parser.add_argument("--left-label", default="NSD")
    parser.add_argument("--right-label", default="baseline")
    parser.add_argument("--blocks", nargs="+", type=int, default=[25, 50, 100],
                        help="Moving-block lengths; 50 is primary by protocol.")
    parser.add_argument("--primary-block", type=int, default=50)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--require-files", nargs="*", default=None,
                        help="Exact JSON filenames required in both inputs; rejects partial pairing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.blocks or any(block <= 0 for block in args.blocks):
        raise SystemExit("--blocks must contain positive integers")
    if args.primary_block not in args.blocks:
        raise SystemExit("--primary-block must be included in --blocks")

    names = sorted(p.name for p in args.left_dir.glob("*.json") if (args.right_dir / p.name).is_file())
    if not names:
        raise SystemExit("no shared per-query JSON files")
    if args.require_files is not None and set(names) != set(args.require_files):
        raise SystemExit(
            "paired input files differ from the required complete set: "
            f"found={names}, required={sorted(args.require_files)}")

    all_pairs, per_sequence = [], {}
    left_hits = right_hits = left_only = right_only = total = 0
    for name in names:
        left = load_records(args.left_dir / name)
        right = load_records(args.right_dir / name)
        if set(left) != set(right):
            raise SystemExit(f"query-index mismatch: {name}")
        idx = sorted(left)
        a = np.array([left[i] for i in idx], dtype=np.int8)
        b = np.array([right[i] for i in idx], dtype=np.int8)
        lo = int(((a == 1) & (b == 0)).sum())
        ro = int(((a == 0) & (b == 1)).sum())
        per_sequence[name.removesuffix(".json")] = {
            "n_queries": len(a), "left_r1": float(a.mean()), "right_r1": float(b.mean()),
            "left_only": lo, "right_only": ro, "mcnemar_p": mcnemar_exact(lo, ro),
        }
        all_pairs.append((a, b))
        left_hits += int(a.sum()); right_hits += int(b.sum())
        left_only += lo; right_only += ro; total += len(a)

    by_block = {}
    for block in args.blocks:
        draws = bootstrap(all_pairs, block, args.reps)
        ci = np.percentile(draws, [2.5, 97.5])
        lower = (int((draws <= 0).sum()) + 1) / (args.reps + 1)
        upper = (int((draws >= 0).sum()) + 1) / (args.reps + 1)
        by_block[str(block)] = {
            "bootstrap_ci95_pp": [float(100 * ci[0]), float(100 * ci[1])],
            "bootstrap_p_two_sided": min(1.0, 2 * min(lower, upper)),
            "is_primary_block": block == args.primary_block,
        }
    primary = by_block[str(args.primary_block)]
    report = {
        "left": args.left_label, "right": args.right_label,
        "primary_block": args.primary_block, "sensitivity_blocks": args.blocks,
        "reps": args.reps,
        "n_queries": total, "left_r1": left_hits / total, "right_r1": right_hits / total,
        "gap_pp": 100 * (left_hits - right_hits) / total,
        "bootstrap_ci95_pp": primary["bootstrap_ci95_pp"],
        "bootstrap_p_two_sided": primary["bootstrap_p_two_sided"],
        "bootstrap_by_block": by_block,
        "mcnemar_pooled": {"left_only": left_only, "right_only": right_only,
                             "p": mcnemar_exact(left_only, right_only)},
        "per_sequence": per_sequence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"saved {args.output}; gap={report['gap_pp']:+.2f}pp")


if __name__ == "__main__":
    main()
