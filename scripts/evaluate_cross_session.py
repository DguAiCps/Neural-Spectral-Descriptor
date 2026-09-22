#!/usr/bin/env python3
"""Cross-session (database vs. query) place-recognition evaluation.

Consumes per-sequence operating caches that were already built by the
checkpoint evaluators (scripts/evaluate_nclt_checkpoint.py /
evaluate_mulran_checkpoint.py: fields ``descriptors``, ``poses``,
``scan_ids``, ...) or the 416D retrieval-key dumps from
scripts/_dump_final416_seed1.py (fields ``emb416``, ``poses``).
Torch-free: runs on host CPU, never re-runs the encoder.

Protocol examples:
  MulRan:  --db-cache DCC01.npz DCC02.npz --query-cache DCC03.npz
  NCLT (RING/RING++ convention, github.com/lus6-Jenny/RING README:
  database 2012-02-04 sampled every 20 m, query 2012-03-17 every 5 m;
  their example uses a 20 m revisit threshold, the RING++ paper 10 m):
           --db-cache nclt_2012-02-04.npz --query-cache nclt_2012-03-17.npz \
           --db-spacing 20 --query-spacing 5

Success = any of the top-k cosine neighbours lies within --threshold metres
of the query pose. Queries with no database pose within the threshold are
excluded from the recall denominator and reported separately.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

KEY_FIELD_PRIORITY = ("emb416", "descriptors")


def _normalize(x: np.ndarray) -> np.ndarray:
    # Same convention as run_kitti_operating_point._normalize (not imported:
    # that module requires torch and this script must stay torch-free).
    x = x.astype(np.float32).copy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def _positions(poses: np.ndarray, planar: bool) -> np.ndarray:
    poses = np.asarray(poses)
    if poses.ndim == 3 and poses.shape[1:] == (4, 4):
        pos = poses[:, :3, 3].astype(np.float64)
    elif poses.ndim == 2 and poses.shape[1] in (2, 3):
        pos = np.zeros((len(poses), 3), dtype=np.float64)
        pos[:, : poses.shape[1]] = poses
    else:
        raise ValueError(f"Unsupported poses shape {poses.shape} (expected (N,4,4) or (N,2|3))")
    if planar:
        pos = pos.copy()
        pos[:, 2] = 0.0
    return pos


def _spacing_indices(positions: np.ndarray, spacing: Optional[float]) -> np.ndarray:
    """Greedy trajectory-order downsampling: keep the first pose, then every
    pose at least ``spacing`` metres from the last kept one."""
    if spacing is None or spacing <= 0 or len(positions) == 0:
        return np.arange(len(positions), dtype=np.int64)
    keep = [0]
    for i in range(1, len(positions)):
        if np.linalg.norm(positions[i] - positions[keep[-1]]) >= spacing:
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def _load_cache(
    path: Path,
    key_field: str,
    spacing: Optional[float],
    planar: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    npz = np.load(path)
    if key_field == "auto":
        field = next((f for f in KEY_FIELD_PRIORITY if f in npz.files), None)
        if field is None:
            raise KeyError(f"{path}: none of {KEY_FIELD_PRIORITY} found (fields: {npz.files})")
    else:
        field = key_field
    keys = np.asarray(npz[field], dtype=np.float32)
    positions = _positions(npz["poses"], planar)
    if len(keys) != len(positions):
        raise ValueError(f"{path}: {field} has {len(keys)} rows but poses has {len(positions)}")
    kept = _spacing_indices(positions, spacing)
    info = {
        "path": str(path),
        "key_field": field,
        "key_dim": int(keys.shape[1]),
        "n_raw": int(len(keys)),
        "n_kept": int(len(kept)),
    }
    return keys[kept], positions[kept], info


def _load_group(
    paths: List[Path],
    key_field: str,
    spacing: Optional[float],
    planar: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    keys_list, pos_list, infos = [], [], []
    for path in paths:
        keys, positions, info = _load_cache(path, key_field, spacing, planar)
        keys_list.append(keys)
        pos_list.append(positions)
        infos.append(info)
        print(f"loaded {path}: field={info['key_field']} dim={info['key_dim']} "
              f"n={info['n_raw']} -> {info['n_kept']} after spacing", flush=True)
    dims = {k.shape[1] for k in keys_list}
    if len(dims) != 1:
        raise ValueError(f"Key dimension mismatch across caches: {sorted(dims)}")
    return np.concatenate(keys_list), np.concatenate(pos_list), infos


def evaluate(
    db_keys: np.ndarray,
    db_pos: np.ndarray,
    q_keys: np.ndarray,
    q_pos: np.ndarray,
    thresholds: List[float],
    k_values: List[int],
    top_k: int,
    chunk: int = 512,
) -> Dict[str, Dict]:
    db_normed = _normalize(db_keys)
    q_normed = _normalize(q_keys)
    top_k = min(top_k, len(db_keys))

    stats = {t: {"n_matchable": 0, "n_no_match": 0, "hits": {k: 0 for k in k_values}}
             for t in thresholds}
    for start in range(0, len(q_keys), chunk):
        stop = min(start + chunk, len(q_keys))
        dists = np.linalg.norm(q_pos[start:stop, None, :] - db_pos[None, :, :], axis=2)
        sims = q_normed[start:stop] @ db_normed.T
        part = np.argpartition(-sims, top_k - 1, axis=1)[:, :top_k]
        order = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1), axis=1
        )
        for i in range(stop - start):
            top_dists = dists[i][order[i]]
            min_dist = dists[i].min()
            for t in thresholds:
                if min_dist >= t:
                    stats[t]["n_no_match"] += 1
                    continue
                stats[t]["n_matchable"] += 1
                for k in k_values:
                    if np.any(top_dists[:k] < t):
                        stats[t]["hits"][k] += 1

    results = {}
    for t in thresholds:
        s = stats[t]
        n_total = s["n_matchable"] + s["n_no_match"]
        entry = {
            "n_queries_total": n_total,
            "n_matchable": s["n_matchable"],
            "n_no_match": s["n_no_match"],
            "frac_no_match": s["n_no_match"] / max(n_total, 1),
        }
        for k in k_values:
            entry[f"R@{k}"] = s["hits"][k] / max(s["n_matchable"], 1)
        results[f"threshold_{t:g}m"] = entry
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-cache", nargs="+", required=True, type=Path,
                        help="Operating-cache npz file(s) forming the database")
    parser.add_argument("--query-cache", nargs="+", required=True, type=Path,
                        help="Operating-cache npz file(s) forming the query set")
    parser.add_argument("--key-field", default="auto",
                        help="npz array holding the retrieval key "
                             "(default auto: emb416 if present, else descriptors)")
    parser.add_argument("--db-spacing", type=float, default=None,
                        help="Spatially downsample each DB cache to this pose spacing "
                             "in metres (default: none; RING++ NCLT uses 20)")
    parser.add_argument("--query-spacing", type=float, default=None,
                        help="Spatially downsample each query cache to this pose spacing "
                             "in metres (default: none; RING++ NCLT uses 5)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Retrieval depth (must cover the largest reported k)")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Revisit distance threshold in metres (a 20 m variant "
                             "is always reported alongside)")
    parser.add_argument("--planar", action="store_true",
                        help="Use xy-only pose distance (guards against per-session "
                             "z conventions; default 3D, matching the intra-session "
                             "evaluators)")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "results/cross_session_eval.json")
    args = parser.parse_args()

    db_keys, db_pos, db_infos = _load_group(
        args.db_cache, args.key_field, args.db_spacing, args.planar)
    q_keys, q_pos, q_infos = _load_group(
        args.query_cache, args.key_field, args.query_spacing, args.planar)
    if db_keys.shape[1] != q_keys.shape[1]:
        raise ValueError(
            f"DB key dim {db_keys.shape[1]} != query key dim {q_keys.shape[1]}")

    thresholds = sorted({float(args.threshold), 20.0})
    k_values = [1, 5]
    results = evaluate(db_keys, db_pos, q_keys, q_pos, thresholds, k_values, args.top_k)

    payload = {
        "settings": {
            "command": shlex.join(sys.argv),
            "db_caches": db_infos,
            "query_caches": q_infos,
            "key_field": args.key_field,
            "db_spacing": args.db_spacing,
            "query_spacing": args.query_spacing,
            "top_k": args.top_k,
            "thresholds": thresholds,
            "k_values": k_values,
            "planar": args.planar,
            "n_db": int(len(db_keys)),
            "n_query": int(len(q_keys)),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
