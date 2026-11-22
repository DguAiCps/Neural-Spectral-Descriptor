#!/usr/bin/env python3
"""Evaluate NSD on encoded and refined sequences, or summarise finished runs.

match:      per-sequence .npz (scan_ids, timestamps, poses, poses_gt, d, z, z_refined) -> per-query ledger
            (query, returned, score, min_past_dist, returned_dist) -> per-sequence and 6-city macro metrics.
summarize:  several run folders (one per training seed) -> mean and SD of the 6-city macro metrics.

    python tools/evaluate.py match --inputs SEQ_DIR --out RUN_DIR [--device cuda]
    python tools/evaluate.py summarize RUN_DIR_SEED4 RUN_DIR_SEED5 RUN_DIR_SEED6

A sequence file is named '<dataset>_<sequence>.npz' (e.g. kitti_00.npz, mcd_kth_day_06.npz); the sequence id
used in the city grouping (assets/test_cities.json) is '<dataset>/<sequence>'.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Single-threaded BLAS: candidate scores are inner products, and exact ties at the top-100 boundary
# depend on the summation order.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from nsd import deterministic, matching, metrics  # noqa: E402


def sequence_id(path: Path) -> str:
    return path.stem.replace("_", "/", 1)


def evaluate_sequence(path: Path, assets: dict, device: str) -> dict:
    """Method (no ground truth) -> ledger, then ground-truth columns for evaluation."""
    with np.load(path) as s:
        data = {k: s[k] for k in ("d", "z", "z_refined", "poses", "poses_gt")}
    ledger = matching.match_sequence(data["d"], data["z"], data["z_refined"], data["poses"], assets, device)
    gt = data["poses_gt"][:, :3, 3]
    ledger["min_past_dist"] = metrics.min_past_distance(gt, ledger["query"])
    ledger["returned_dist"] = metrics.returned_distance(gt, ledger["query"], ledger["returned"])
    return ledger


def match(args) -> None:
    assets = matching.load_assets(args.assets)
    cities = json.loads((Path(args.assets) / "test_cities.json").read_text())
    wanted = {s for v in cities.values() for s in v}
    out = Path(args.out)
    (out / "ledgers").mkdir(parents=True, exist_ok=True)
    per = {}
    for path in sorted(Path(args.inputs).glob("*.npz")):
        seq = sequence_id(path)
        if seq not in wanted:
            continue
        ledger = evaluate_sequence(path, assets, args.device)
        np.savez_compressed(out / "ledgers" / path.name, **ledger)
        per[seq] = metrics.sequence_metrics(ledger["min_past_dist"], ledger["returned_dist"], ledger["score"])
        print(seq, per[seq], flush=True)
    missing = sorted(wanted - set(per))
    if missing:
        print("warning: missing sequences, macro is over the available ones:", missing)
    result = dict(metrics.macro(per, cities), sequences=per, missing=missing)
    (out / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
    print("6-city macro %", {k: round(100 * v, 2) for k, v in result["macro"].items()})


def summarize(args) -> None:
    macros = [json.loads((Path(r) / "metrics.json").read_text()) for r in args.runs]
    stats = metrics.seed_mean(macros)
    for k, v in stats.items():
        sd = f" +- {100 * v['sd']:.2f}" if v["sd"] is not None else ""
        print(f"{k:8s} {100 * v['mean']:.2f}{sd}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    m = sub.add_parser("match")
    m.add_argument("--inputs", required=True, help="folder of per-sequence .npz files")
    m.add_argument("--out", required=True, help="output folder (ledgers/ and metrics.json)")
    m.add_argument("--assets", default=str(PKG / "assets"))
    m.add_argument("--device", default="cuda")
    m.set_defaults(fn=match)
    s = sub.add_parser("summarize")
    s.add_argument("runs", nargs="+", help="run folders written by 'match', one per seed")
    s.set_defaults(fn=summarize)
    args = p.parse_args()
    deterministic()
    args.fn(args)


if __name__ == "__main__":
    main()
