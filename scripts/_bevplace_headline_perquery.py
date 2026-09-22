#!/usr/bin/env python3
"""Regenerate Table-1-gated BEVPlace++ per-query records for paired tests.

The Table-1 BEVPlace++ row was evaluated on heterogeneous cache lineages:
the common 056e0a02 cache for KITTI/NCLT/HeLiPR and 33919e6e for MulRan.
This script fixes those paths explicitly, uses the fine-tuned checkpoint, and
only writes outputs after every per-sequence R@1 matches the frozen Table-1
reference within rounding tolerance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", REPO / "data"))
DEFAULT_WEIGHTS = REPO / "baselines/weights/bevplace_finetune.pth"
TOL = 5e-4  # Table-1 values are reported to three decimals.

# Canonical display name, dataset type, sequence, exact cache lineage, Table-1 R@1.
CASES = [
    ("KITTI_00", "kitti", "00", "data/preprocessed/cache_056e0a02_kitti_val_00.npz", 0.9953),
    ("KITTI_05", "kitti", "05", "data/preprocessed/cache_056e0a02_kitti_val_05.npz", 0.9814),
    ("KITTI_08", "kitti", "08", "data/preprocessed/cache_056e0a02_kitti_val_08.npz", 0.9234),
    ("NCLT_2012-01-08", "nclt", "2012-01-08", "data/preprocessed/cache_056e0a02_nclt_val_2012-01-08.npz", 0.8969),
    ("NCLT_2013-01-10", "nclt", "2013-01-10", "data/preprocessed/cache_056e0a02_nclt_val_2013-01-10.npz", 0.9061),
    ("HeLiPR_Town01", "helipr", "Town01", "data/preprocessed/cache_056e0a02_helipr_val_Town01.npz", 0.7421),
    ("MulRan_DCC03", "mulran", "DCC03", "data/preprocessed/cache_33919e6e_mulran_val_DCC03.npz", 0.7265),
    ("MulRan_KAIST03", "mulran", "KAIST03", "data/preprocessed/cache_33919e6e_mulran_val_KAIST03.npz", 0.9955),
    ("MulRan_Riverside03", "mulran", "Riverside03", "data/preprocessed/cache_33919e6e_mulran_val_Riverside03.npz", 0.8659),
]


def make_loader(kind: str, sequence: str):
    if kind == "kitti":
        from data.kitti_loader import KITTILoader
        return KITTILoader(str(DATA_ROOT / "kitti/dataset"), sequence, lazy_load=True)
    if kind == "nclt":
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(str(DATA_ROOT / "nclt"), sequence, lazy_load=True)
    if kind == "helipr":
        from data.helipr_loader import HeLiPRLoader
        return HeLiPRLoader(str(DATA_ROOT / "helipr" / sequence / sequence), lazy_load=True)
    if kind == "mulran":
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(str(DATA_ROOT / "mulran"), sequence, lazy_load=True)
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "results/per_query_baselines_regen/bevplace")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.glob("*.json")) and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output_dir}; pass --force after inspection")
    if not args.weights.is_file():
        raise SystemExit(f"missing BEVPlace++ checkpoint: {args.weights}")

    os.environ["NSD_BEVPLACE_WEIGHTS"] = str(args.weights)
    from baselines.bevplace import BEVPlaceBaseline
    from baselines.eval_utils import compute_recall_multi_k

    encoder = BEVPlaceBaseline()
    if not encoder.is_available():
        raise SystemExit("BEVPlace++ dependencies or weights are unavailable")

    pending, summary = {}, {}
    for name, kind, sequence, rel_cache, target in CASES:
        cache_path = REPO / rel_cache
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        with np.load(cache_path, allow_pickle=False) as cache:
            scan_ids = cache["scan_ids"].astype(np.int64)
            poses = cache["poses"].astype(np.float64)
        print(f"[{name}] encoding {len(scan_ids)} keyframes from {cache_path}", flush=True)
        loader = make_loader(kind, sequence)
        desc = []
        for pos, scan_id in enumerate(scan_ids):
            desc.append(encoder.encode(loader[int(scan_id)]["points"]))
            if (pos + 1) % 500 == 0:
                print(f"  {pos + 1}/{len(scan_ids)}", flush=True)
        records: list[dict] = []
        recalls, n_queries = compute_recall_multi_k(
            np.asarray(desc, dtype=np.float32), poses, k_values=[1, 5, 10],
            distance_threshold=5.0, skip_frames=30, per_query_records=records)
        r1 = float(recalls[1])
        mismatch = abs(r1 - target)
        print(f"  R@1={r1:.4f}; Table-1={target:.4f}; q={n_queries}", flush=True)
        if mismatch > TOL:
            raise SystemExit(
                f"ABORT {name}: R@1={r1:.4f} differs from Table-1 {target:.4f} "
                f"by {mismatch:.4f} (> {TOL})")
        pending[name] = {
            "method": "bevplace",
            "dataset": name,
            "distance_threshold_m": 5.0,
            "skip_frames": 30,
            "n_queries": n_queries,
            "recall_at_1": r1,
            "recall_at_5": float(recalls[5]),
            "recall_at_10": float(recalls[10]),
            "table1_target_r1": target,
            "cache": rel_cache,
            "records": records,
        }
        summary[name] = {"r1": r1, "table1_target_r1": target,
                         "n_queries": n_queries, "cache": rel_cache}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, record in pending.items():
        (args.output_dir / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
    manifest = {
        "command": sys.argv,
        "data_root": str(DATA_ROOT),
        "weights": str(args.weights),
        "table1_tolerance": TOL,
        "sequences": summary,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"WROTE {args.output_dir} (all Table-1 gates passed)", flush=True)


if __name__ == "__main__":
    main()
