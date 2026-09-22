#!/usr/bin/env python3
"""pd2026 baseline eval: BEVPlace++ on the frozen 42-sequence pd test roster.

Protocol is identical to the NSD pd stack: keyframes + poses come from the
committed pd2026 operating caches (scripts/_pd_test_cache_manifest.json),
intra-session revisit queries with distance_threshold 5 m / skip_frames 30,
R@1/R@5/R@10 (baselines/eval_utils.compute_recall_multi_k — the same scorer
the paper's Table-1 BEVPlace++ row used).

Descriptors are re-encoded from raw point clouds at the cached scan_ids with
baselines/bevplace.py (official vendored REIN, 200x200 BEV, max_range 50 m).

Memory discipline: one heavy sequence per process — drive with
    --seqs sensor/seq [sensor/seq ...]
one subprocess each (cgroup ~28 GB). Point clouds are loaded one keyframe at
a time (never the whole sequence).

Usage (inside docker ring_baseline):
    python3 scripts/_pdb_bevplace_eval.py \
        --weights baselines/weights/bevplace_pd2026.pth \
        --seqs mulran/DCC03
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
DATA_ROOT = Path(os.environ.get("NSD_DATA_ROOT", "/workspace/data"))
MANIFEST = REPO / "scripts/_pd_test_cache_manifest.json"


def make_loader(sensor: str, seq: str):
    if sensor == "kitti":
        from data.kitti_loader import KITTILoader
        return KITTILoader(str(DATA_ROOT / "kitti/dataset"), seq, lazy_load=True)
    if sensor == "nclt":
        from data.nclt_loader import NCLTLoader
        return NCLTLoader(str(DATA_ROOT / "nclt"), seq, lazy_load=True)
    if sensor == "helipr":
        from data.helipr_loader import HeLiPRLoader
        seq_path = DATA_ROOT / "helipr" / seq
        if not (seq_path / "LiDAR").is_dir():
            seq_path = seq_path / seq  # legacy double-nested layout
        return HeLiPRLoader(str(seq_path), lazy_load=True)
    if sensor == "mulran":
        from data.mulran_loader import MulRanLoader
        return MulRanLoader(str(DATA_ROOT / "mulran"), seq, lazy_load=True)
    if sensor == "kitti360":
        from data.kitti360_loader import Kitti360Loader
        return Kitti360Loader(str(DATA_ROOT / "kitti360"), seq, lazy_load=True)
    if sensor == "mcd":
        from data.mcd_loader import MCDLoader
        return MCDLoader(str(DATA_ROOT / "mcd_extracted"), seq, lazy_load=True)
    raise ValueError(sensor)


def sha256_head(path: Path, n_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def eval_sequence(encoder, sensor: str, seq: str, cache_rel: str) -> dict:
    from baselines.eval_utils import compute_recall_multi_k

    cache_path = REPO / cache_rel
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as cache:
        scan_ids = cache["scan_ids"].astype(np.int64)
        poses = cache["poses"].astype(np.float64)

    loader = make_loader(sensor, seq)
    desc = []
    encode_s = []
    print(f"[{sensor}/{seq}] encoding {len(scan_ids)} keyframes "
          f"from {cache_rel}", flush=True)
    for pos, sid in enumerate(scan_ids):
        pts = loader[int(sid)]["points"]
        t0 = time.perf_counter()
        desc.append(encoder.encode(pts))
        encode_s.append(time.perf_counter() - t0)
        if (pos + 1) % 500 == 0:
            print(f"  {pos + 1}/{len(scan_ids)} "
                  f"({np.mean(encode_s[-500:]) * 1e3:.1f} ms/kf)", flush=True)

    recalls, n_queries = compute_recall_multi_k(
        np.asarray(desc, dtype=np.float32), poses, k_values=[1, 5, 10],
        distance_threshold=5.0, skip_frames=30)
    return {
        "method": "bevplace",
        "sensor": sensor,
        "sequence": seq,
        "protocol": {"distance_threshold_m": 5.0, "skip_frames": 30,
                     "intra_session": True},
        "cache": cache_rel,
        "n_keyframes": int(len(scan_ids)),
        "n_queries": int(n_queries),
        "recall_at_1": float(recalls[1]),
        "recall_at_5": float(recalls[5]),
        "recall_at_10": float(recalls[10]),
        "encode_ms_per_kf": float(np.mean(encode_s) * 1e3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=REPO / "baselines/weights/bevplace_pd2026.pth")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "results/pd2026_baselines")
    parser.add_argument("--seqs", nargs="+", default=None,
                        help="subset like mulran/DCC03 mcd/tuhh_night_09 "
                             "(default: all 42)")
    parser.add_argument("--force", action="store_true",
                        help="re-run sequences whose output JSON exists")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"missing BEVPlace++ checkpoint: {args.weights}")
    os.environ["NSD_BEVPLACE_WEIGHTS"] = str(args.weights)
    from baselines.bevplace import BEVPlaceBaseline

    encoder = BEVPlaceBaseline()
    if not encoder.is_available():
        raise SystemExit("BEVPlace++ dependencies or weights unavailable")

    roster = json.load(open(MANIFEST))["sequences"]
    wanted = set(args.seqs) if args.seqs else None
    todo = [(s["sensor"], s["sequence"], s["expected"]["operating_cache"])
            for s in roster
            if wanted is None or f"{s['sensor']}/{s['sequence']}" in wanted]
    if wanted is not None and len(todo) != len(wanted):
        have = {f"{a}/{b}" for a, b, _ in todo}
        raise SystemExit(f"unknown --seqs entries: {sorted(wanted - have)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_id = sha256_head(args.weights)
    for sensor, seq, cache_rel in todo:
        out_path = args.output_dir / f"bevplace_{sensor}_{seq}.json"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path} exists", flush=True)
            continue
        record = eval_sequence(encoder, sensor, seq, cache_rel)
        record["weights"] = str(args.weights)
        record["weights_sha256_1mb"] = weights_id
        out_path.write_text(json.dumps(record, indent=1) + "\n")
        print(f"  R@1={record['recall_at_1']:.4f} R@5={record['recall_at_5']:.4f} "
              f"q={record['n_queries']} "
              f"({record['encode_ms_per_kf']:.1f} ms/kf) -> {out_path}",
              flush=True)


if __name__ == "__main__":
    main()
