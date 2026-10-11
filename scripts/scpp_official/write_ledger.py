#!/usr/bin/env python3
"""pr-dump-v1 ledger (and aggregate) for the official SC++ MATLAB run.

Turns the per-query CSV of scpp_official_match.m into the same npz schema as
scripts/_causal_scpp_pr.py (arrays query_idx / j_star / score_scpp /
top1_geo_dist / min_past_gt_dist / n_candidates + json meta), so
scripts/_pdv2_c3.py cmd_aggregate_baselines reads it unchanged.

  score_scpp     = -(official SC distance of the accepted top-1)  (higher =
                   accept), NaN when the official candidate loop accepted
                   nothing (NaN distance); j_star = -1, top1_geo_dist = NaN then
  top1_geo_dist  = 3D GT distance query -> j_star (poses_gt)
  min_past_gt_dist = min 3D GT distance query -> any frame <= t - 30
  n_candidates   = 1 (k = 1 retrieval-key candidate, paper Table II)

Host-side, numpy only:
  python3 scripts/scpp_official/write_ledger.py --seq kitti/07 --variant apc \
      --work W --csv C [--out-root results/pdv2_scpp_official]
  python3 scripts/scpp_official/write_ledger.py --aggregate --variant apc
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

OUT_ROOT = REPO / "results/pdv2_scpp_official"
CACHE_DIR = {"odom": REPO / "data/preprocessed_v2_odom",
             "gt": REPO / "data/preprocessed_v2"}
MIN_INDEX_GAP = 30
OFFICIAL = {
    "repo": "https://github.com/gisbi-kim/scancontext_tro",
    "commit": "b463bb7",
    "local_clone": "_external/scancontext_tro (unmodified)",
    "paper": "Kim, Choi, Kim, Scan Context++, IEEE T-RO 2022, arXiv:2109.13494",
}
VARIANTS = {
    "apc": "A-PC: polar context 20x60, 80 m, + 2 lateral (y -/+2 m) root-shift "
           "augmentations in one k-d tree (paper Table II / Sec. V-D.1)",
    "pc": "PC: polar context 20x60, 80 m, no augmentation",
    "cc": "CC: Cartesian context 40x40, x +-100 m (5 m) / y +-40 m (2 m), no "
          "augmentation",
    "acc": "A-CC: CC + double-flip copy in one k-d tree (paper Sec. V-D.2; the "
           "flip is not in the released code)",
}


def _cache_path(cond: str, sensor: str, q: str) -> Path:
    hits = glob.glob(str(CACHE_DIR[cond] / f"cache_*_{sensor}_test_{q}.npz"))
    assert len(hits) == 1, hits
    return Path(hits[0])


SC_PARAMS = {
    "pc": dict(num_ring=20, num_sector=60, max_range=80.0, grid_step=0.5,
               min_points_per_bin=5, ring_key="sc2invkey (row mean)",
               aligning_key="sc2vkey (column mean)"),
    "cc": dict(num_x=40, num_y=40, axis_unit=[5.0, 2.0], axis_range=[100.0, 40.0],
               grid_step=0.5, ring_key="sc2invkey (row mean)",
               aligning_key="sc2vkey (column mean)"),
}
SC_PARAMS["apc"] = dict(SC_PARAMS["pc"], n_aug=3, root_shift_y_m=[0.0, -2.0, 2.0])
SC_PARAMS["acc"] = dict(SC_PARAMS["cc"], n_aug=2, augmentation="flip(flip(cc,1),2)")


def write(seq: str, variant: str, work: Path, csv: Path, out_root: Path,
          cond: str):
    sensor, q = seq.split("/")
    side = json.loads(Path(f"{csv}.json").read_text())
    assert side["variant"] == variant, (side["variant"], variant)
    params = dict(SC_PARAMS[variant], lidar_height=side["lidar_height"],
                  search_ratio=side["search_ratio"], k_candidates=1,
                  matlab_version=side["matlab_version"])
    info = json.loads((work / "seq_info.json").read_text())
    assert (info["sensor"], info["sequence"], info["cond"]) == (sensor, q, cond)
    cache = _cache_path(cond, sensor, q)
    assert str(cache.relative_to(REPO)) == info["operating_cache"], \
        (cache, info["operating_cache"])
    with np.load(cache) as c:
        poses = (c["poses_gt"] if "poses_gt" in c.files
                 else c["poses"]).astype(np.float64)
    n = len(poses)
    assert n == info["n_frames"], (n, info["n_frames"])
    positions = poses[:, :3, 3]

    raw = np.genfromtxt(csv, delimiter=",", names=True, dtype=np.float64)
    queries = raw["query_idx"].astype(np.int64)
    assert (queries == np.arange(MIN_INDEX_GAP, n)).all(), "query set != all t >= 30"
    j_star = raw["j_star"].astype(np.int64)
    cand = raw["cand_frame"].astype(np.int64)
    dist = raw["cand_dist"]
    acc = j_star >= 0
    assert (j_star[acc] == cand[acc]).all()
    assert np.isfinite(dist[acc]).all() and np.isnan(dist[~acc]).all()
    assert (cand <= queries - MIN_INDEX_GAP).all(), "candidate beyond the causal prefix"

    score = np.where(acc, -dist, np.nan)
    top1 = np.full(len(queries), np.nan)
    top1[acc] = np.linalg.norm(positions[j_star[acc]] - positions[queries[acc]], axis=1)
    min_past = np.empty(len(queries))
    for k, t in enumerate(queries):
        min_past[k] = float(np.linalg.norm(
            positions[:t - MIN_INDEX_GAP + 1] - positions[t], axis=1).min())
    arrays = dict(query_idx=queries, j_star=j_star, score_scpp=score,
                  top1_geo_dist=top1, min_past_gt_dist=min_past,
                  n_candidates=np.ones(len(queries), dtype=np.int64))

    meta = {
        "schema": "pr-dump-v1",
        "method": "scpp",
        "sensor": sensor, "sequence": q,
        "n_frames": n, "n_queries": int(len(queries)),
        "min_index_gap": MIN_INDEX_GAP,
        "n_coarse": 1,
        "sc_params": params,
        "limit": 0,
        "variant": variant,
        "variant_desc": VARIANTS[variant],
        "official_code": OFFICIAL,
        "n_no_match": int((~acc).sum()),
        "score_negation": (
            "score_scpp = -dist, dist = official sc_dist_fast_with_argalign of "
            "the k=1 retrieval-key candidate at the aligning-key shift "
            "(stored NEGATED: higher = accept); NaN = official candidate loop "
            "accepted nothing (NaN distance)"),
        "artifacts": {
            "operating_cache": info["operating_cache"],
            "operating_cache_sha256": info["operating_cache_sha256"],
            "weights": None,
        },
        "code": {
            "export": "scripts/scpp_official/export_keyframes.py (container)",
            "descriptor": "scripts/scpp_official/scpp_official_describe.m -> "
                          "official ptcloud2polarcontextAug.m / ptcloud2cartcontext.m",
            "matching": "scripts/scpp_official/scpp_official_match.m -> official "
                        "sc2invkey, sc2vkey, alignUsingSectorKey, "
                        "sc_dist_fast_with_argalign; MATLAB createns/knnsearch",
            "ledger": "scripts/scpp_official/write_ledger.py",
            "raw_csv": str(csv.relative_to(REPO)) if csv.is_relative_to(REPO) else str(csv),
        },
        "protocol": (
            "causal all-frames ledger on split-v2 odometry keyframes: per query "
            "t >= 30 the map is j <= t - 30; retrieval-key k-d tree rebuilt "
            "per query over that prefix (all augmentation copies), k = 1, "
            "aligning-key pre-alignment, official SC distance; "
            "docs/CAUSAL_PR_PREREGISTRATION.md §1.1/§1.2/§1.5"),
        "deviation_from_intra": "n/a (no place-separated official SC++ run); "
                                "see deviation_from_official",
        "deviation_from_official": (
            "see logs/baseline_official_20261011/scpp.md: own keyframes and "
            "30-frame exclusion (protocol), exact-prefix tree per query, both "
            "PC and CC from one cloud, R2026a pcdownsample"),
        "timing": {k: side[k] for k in ("desc_pc_aug_ms_per_kf",
                   "desc_cc_ms_per_kf", "match_ms_per_query", "n_points_mean")},
    }
    out_dir = out_root / variant / cond
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"dump_scpp_{sensor}_{q}.npz"
    np.savez_compressed(out, meta=json.dumps(meta), **arrays)
    print(f"LEDGER {out.relative_to(REPO)} n_queries={len(queries)} "
          f"no_match={int((~acc).sum())}", flush=True)


def aggregate(variant: str, cond: str, out_root: Path):
    """The existing split-v2 baseline aggregation, pointed at this folder."""
    import _pdv2_c3 as c3
    c3.OUT_ROOT = out_root / variant
    c3.BASELINES = {"scpp": "score_scpp"}
    c3.cmd_aggregate_baselines(cond)
    p = out_root / variant / cond / "pr_summary_baselines.json"
    s = json.loads(p.read_text())
    s["_meta"].pop("seqot", None)
    s["_meta"].update(variant=variant, variant_desc=VARIANTS[variant],
                      official_code=OFFICIAL,
                      aggregator="scripts/_pdv2_c3.py:cmd_aggregate_baselines")
    p.write_text(json.dumps(s, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seq", metavar="SENSOR/SEQ")
    ap.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    ap.add_argument("--cond", default="odom", choices=tuple(CACHE_DIR))
    ap.add_argument("--work")
    ap.add_argument("--csv")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.variant, a.cond, Path(a.out_root).resolve())
        return
    write(a.seq, a.variant, Path(a.work).resolve(), Path(a.csv).resolve(),
          Path(a.out_root).resolve(), a.cond)


if __name__ == "__main__":
    main()
