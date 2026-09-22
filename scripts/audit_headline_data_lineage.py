#!/usr/bin/env python3
"""Audit cache/raw lineage prerequisites for headline paired experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", REPO / "data"))
NCLT_DATES = ("2012-01-08", "2013-01-10")
MULRAN = ("DCC03", "KAIST03", "Riverside03")
EXPECTED_QUERIES = {"DCC03": 2344, "KAIST03": 2909, "Riverside03": 2796}
TIMESTAMP_TOLERANCE_NS = 256


def fingerprint(path: Path) -> str:
    with np.load(path, mmap_mode="r") as z:
        a = z["fft_magnitudes"]
        return hashlib.sha256(np.ascontiguousarray(a[:4]).tobytes()).hexdigest()


def query_count(poses: np.ndarray) -> int:
    pos = poses[:, :3, 3]
    tree = cKDTree(pos)
    return sum(any(i <= j - 30 for i in tree.query_ball_point(pos[j], 5.0))
               for j in range(30, len(pos)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/supplementary/headline_data_lineage.json")
    args = parser.parse_args()

    report: dict[str, object] = {"data_root": str(DATA_ROOT), "nclt": {}, "mulran": {}}
    for date in NCLT_DATES:
        canonical = REPO / "data/preprocessed_nclt_fix8" / f"nclt_operating_{date}_layout60_stride1.npz"
        candidates = []
        for path in sorted((REPO / "data").glob(f"**/nclt_operating_{date}_layout60_stride1.npz")):
            if path == canonical:
                continue
            with np.load(path, mmap_mode="r") as z:
                candidates.append({"path": str(path.relative_to(REPO)),
                                   "n_keyframes": int(len(z["scan_ids"])),
                                   "scan_id_max": int(z["scan_ids"].max()),
                                   "fft_prefix_sha256": fingerprint(path)})
        report["nclt"][date] = {"canonical_path": str(canonical.relative_to(REPO)),
                                 "canonical_exists": canonical.is_file(),
                                 "candidates": candidates}

    from data.mulran_loader import MulRanLoader
    for sequence in MULRAN:
        cache = REPO / "data/preprocessed_cross_sensor_operating" / f"mulran_operating_{sequence}_layout60_stride1.npz"
        with np.load(cache, mmap_mode="r") as z:
            scan_ids = z["scan_ids"]
            poses = z["poses"]
            cache_timestamps_ns = np.rint(z["timestamps"] * 1e9).astype(np.int64)
            n_q = query_count(poses)
        loader = MulRanLoader(str(DATA_ROOT / "mulran"), sequence, lazy_load=True)
        raw_timestamps_ns = loader.scan_timestamps_ns[scan_ids]
        timestamp_error_ns = np.abs(cache_timestamps_ns - raw_timestamps_ns)
        report["mulran"][sequence] = {
            "cache": str(cache.relative_to(REPO)),
            "cache_keyframes": int(len(scan_ids)),
            "cache_scan_id_max": int(scan_ids.max()),
            "loader_scans": int(len(loader)),
            "scan_ids_in_loader_range": bool(int(scan_ids.max()) < len(loader)),
            "cache_timestamp_vs_raw_max_error_ns": int(timestamp_error_ns.max()),
            "scan_order_matches_raw_within_timestamp_tolerance": bool(
                np.all(timestamp_error_ns <= TIMESTAMP_TOLERANCE_NS)
            ),
            "timestamp_tolerance_ns": TIMESTAMP_TOLERANCE_NS,
            "query_count": n_q,
            "expected_query_count": EXPECTED_QUERIES[sequence],
            "query_count_matches": n_q == EXPECTED_QUERIES[sequence],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
