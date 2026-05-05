"""Compute Recall@1 split by yaw bin from per-query JSON dumps.

Background
----------
The aggregated R@1 in Table 2 averages across all revisit queries within a
sequence. KITTI 08's per-query analysis (Appendix K) shows that 91.5% of its
revisit pairs are *reverse-loop* (|Delta yaw| > 90 deg). If reverse-loop
revisits are systematically harder than forward-loop revisits, the aggregate
R@1 hides this structural asymmetry.

This script splits each per-query record by its |Delta yaw| bin and reports
R@1 separately for forward (|Delta yaw| <= 30 deg), oblique (30-90 deg), and
reverse (> 90 deg) revisits. Comparing per-bin R@1 across sequences operationalises
the failure mode that the unconditioned R@1 averages over.

Inputs
------
- Per-query JSON files produced by `train_multi_dataset.py --validate-only
  --dump-per-query-dir <dir>`. The trainer records `delta_yaw_deg` for each
  query/true-match pair (see src/gnn/trainer.py:_compute_recall_multi_k).

Output
------
JSON: per-dataset {forward, oblique, reverse, all} R@1 + counts.

Usage
-----
    python3 scripts/compute_yaw_recall.py \\
        --per-query-dir results/per_query_v06 \\
        --output results/yaw_recall_split.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


SEQUENCES = [
    "KITTI_00", "KITTI_05", "KITTI_08",
    "NCLT_2012-01-08", "NCLT_2013-01-10",
    "HeLiPR_Town01",
    "MulRan_DCC03", "MulRan_KAIST03", "MulRan_Riverside03",
]

# Baseline eval pipeline uses uppercased prefixes (MULRAN, HELIPR); NSD trainer
# uses MulRan/HeLiPR mixed case. Look up either form when scanning per-query dumps.
NAME_ALIASES = {
    "HeLiPR_Town01": ["HeLiPR_Town01", "HELIPR_Town01"],
    "MulRan_DCC03": ["MulRan_DCC03", "MULRAN_DCC03"],
    "MulRan_KAIST03": ["MulRan_KAIST03", "MULRAN_KAIST03"],
    "MulRan_Riverside03": ["MulRan_Riverside03", "MULRAN_Riverside03"],
}

YAW_BINS = {
    "forward": lambda d: d <= 30.0,
    "oblique": lambda d: (d > 30.0) & (d <= 90.0),
    "reverse": lambda d: d > 90.0,
}


def find_per_query_path(base_dir: Path, dataset: str, method: str = None) -> Path:
    """Locate <base_dir>[/method]/<dataset|alias>.json; return first hit or None."""
    sub = base_dir / method if method else base_dir
    aliases = NAME_ALIASES.get(dataset, [dataset])
    for alias in aliases:
        p = sub / f"{alias}.json"
        if p.exists():
            return p
    return None


def split_by_yaw(records: List[dict]) -> Dict[str, dict]:
    if not records:
        return {bin_name: {"r1": float("nan"), "n": 0} for bin_name in list(YAW_BINS) + ["all"]}
    dyaw = np.abs(np.array([r["delta_yaw_deg"] for r in records]))
    success = np.array([r["success_at_k1"] for r in records])

    out = {}
    for name, fn in YAW_BINS.items():
        mask = fn(dyaw)
        n = int(mask.sum())
        r1 = float(success[mask].mean()) if n > 0 else float("nan")
        out[name] = {"r1": r1, "n": n}
    out["all"] = {"r1": float(success.mean()), "n": int(len(records))}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-query-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="If set, expect per-method subdirs <dir>/<method>/<dataset>.json. "
                         "Otherwise treat <dir>/<dataset>.json as a single-method dump.")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    methods = args.methods if args.methods else [None]
    out_all: Dict[str, Dict[str, dict]] = {}

    for method in methods:
        method_label = method if method else "default"
        out: Dict[str, dict] = {}
        print(f"\n=== method: {method_label} ===")
        for dataset in SEQUENCES:
            path = find_per_query_path(args.per_query_dir, dataset, method)
            if path is None:
                print(f"[skip] {dataset}: missing under {args.per_query_dir}"
                      + (f"/{method}" if method else ""))
                continue
            with open(path) as f:
                data = json.load(f)
            split = split_by_yaw(data["records"])
            out[dataset] = split

            line = f"[ok] {dataset:22s}"
            for bin_name in ["forward", "oblique", "reverse", "all"]:
                v = split[bin_name]
                r1 = v["r1"]
                n = v["n"]
                r1_str = f"{r1:.3f}" if not np.isnan(r1) else "  -- "
                line += f"  {bin_name}={r1_str} (n={n:>4d})"
            print(line)
        out_all[method_label] = out

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"yaw_bins_deg": {"forward": "<=30", "oblique": "30-90",
                                    "reverse": ">90"},
                   "methods": out_all}, f, indent=2)
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
