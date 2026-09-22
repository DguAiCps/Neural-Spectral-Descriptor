#!/usr/bin/env python3
"""Prepare RINGSharp-format training pickles for the pd2026 place-disjoint split.

Derived from scripts/_ringsharp_prepare.py (paper-era 24-sequence split, left
untouched) the same way _pdb_seqot_prepare.py was derived from _seqot_prepare.py.
Builds, under _external/ringsharp_data_pd2026/ (separate staging dir; the
paper-era _external/ringsharp_data/ pickles are never overwritten):
  * train_nsc.pickle  — glnet TrainingTuples over the 16 pd2026 TRAIN sequences
                        (configs/training_place_disjoint.yaml; kitti 01/02/06/07,
                        nclt 5 sessions, helipr Bridge01-03 + Roundabout01-02,
                        mulran Sejong01/02)
  * val_nsc.pickle    — small TrainingTuple pickle (subset of kitti 02); only
                        needed so make_dataloaders(validation=True) constructs
  * manifest.json     — provenance
No side-channel EvaluationSet is built: checkpoint selection happens AFTER
training, on the pd VALIDATION sequences, via our own evaluator (pre-declared
rule: highest sequence-macro 5 m Recall@1 over epochs 1-5, earlier epoch on
ties). test_file stays disabled in the training config.

Positive/non-negative semantics, KITTI pose conjugation and grouping follow
_ringsharp_prepare.py verbatim: positives = same-group frames within 25 m,
non_negatives = within 50 m; NCLT sessions form ONE co-registered campus
group; KITTI / HeLiPR / MulRan are per-sequence groups (their cached pose
frames are origin-normalized per sequence — same documented limitation).

Run inside docker (no GPU needed):
  python scripts/_pdb_ringsharp_prepare.py
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
RINGSHARP = REPO / "_external" / "RINGSharp"
sys.path.insert(0, str(RINGSHARP))
sys.path.insert(1, str(REPO))

from glnet.datasets.base_datasets import TrainingTuple  # noqa: E402

OUT = REPO / "_external" / "ringsharp_data_pd2026"
CONFIG = REPO / "configs" / "training_place_disjoint.yaml"
CACHE_DIR = REPO / "data" / "preprocessed_pd2026"

CACHE_KEYS = {"kitti": "fecdaef7", "nclt": "aff53e59",
              "helipr": "ad22d735", "mulran": "50223b37"}

POS_M, NONNEG_M = 25.0, 50.0            # official RINGSharp NCLT thresholds


def cache_path(dtype, seq):
    return CACHE_DIR / f"cache_{CACHE_KEYS[dtype]}_{dtype}_{seq}.npz"


# KITTI caches store cam0-convention poses; conjugate into a z-up body frame
# (same transform + rationale as _ringsharp_prepare.py).
_R_BC = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
_M_BC = np.eye(4)
_M_BC[:3, :3] = _R_BC


def fix_pose_frame(dtype, poses):
    if dtype != "kitti":
        return poses
    return np.einsum("ij,njk,kl->nil", _M_BC, poses, np.linalg.inv(_M_BC))


def load_seq(dtype, seq):
    with np.load(cache_path(dtype, seq), allow_pickle=True) as c:
        return c["scan_ids"].astype(np.int64), \
            fix_pose_frame(dtype, c["poses"].astype(np.float64)), \
            c["timestamps"].astype(np.float64)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(CONFIG))
    train_list = []
    for ds in cfg["data"]["datasets"]["train"]:
        for seq in ds["sequences"]:
            train_list.append((ds["type"], str(seq)))
    assert len(train_list) == 16, f"expected 16 pd train sequences, got {len(train_list)}"

    seqs = {}
    for dtype, seq in train_list:
        sids, poses, ts = load_seq(dtype, seq)
        seqs[(dtype, seq)] = dict(scan_ids=sids, poses=poses, ts=ts)
        print(f"[{dtype}/{seq}] {len(sids)} keyframes")

    def group_of(dtype, seq):
        return "nclt_campus" if dtype == "nclt" else f"{dtype}_{seq}"

    offset = 0
    order = []
    for key in train_list:
        n = len(seqs[key]["scan_ids"])
        seqs[key]["gid0"] = offset
        order.append(key)
        offset += n
    n_total = offset
    print(f"total {n_total} train keyframes")

    from scipy.spatial import cKDTree
    groups = {}
    for key in order:
        g = group_of(*key)
        s = seqs[key]
        gids = np.arange(s["gid0"], s["gid0"] + len(s["scan_ids"]))
        xy = s["poses"][:, :2, 3]
        groups.setdefault(g, [[], []])
        groups[g][0].append(gids)
        groups[g][1].append(xy)
    for g in groups:
        gids = np.concatenate(groups[g][0])
        xy = np.concatenate(groups[g][1])
        groups[g] = (gids, xy, cKDTree(xy))

    tuples = {}
    n_no_pos = 0
    for dtype, seq in order:
        s = seqs[(dtype, seq)]
        g = group_of(dtype, seq)
        gids, gxy, tree = groups[g]
        xy = s["poses"][:, :2, 3]
        pos_lists = tree.query_ball_point(xy, POS_M)
        nn_lists = tree.query_ball_point(xy, NONNEG_M)
        for i in range(len(s["scan_ids"])):
            gid = s["gid0"] + i
            positives = np.sort(gids[np.asarray(pos_lists[i], dtype=np.int64)])
            positives = positives[positives != gid]
            non_negatives = np.sort(gids[np.asarray(nn_lists[i], dtype=np.int64)])
            if len(positives) == 0:
                n_no_pos += 1
            tuples[gid] = TrainingTuple(
                id=gid, timestamp=int(s["ts"][i]),
                rel_scan_filepath=f"velodyne_sync/{dtype}/{seq}/{int(s['scan_ids'][i]):07d}.pc",
                positives=positives.astype(np.int32),
                non_negatives=non_negatives.astype(np.int32),
                pose=s["poses"][i],
                positives_poses={})
    print(f"tuples: {len(tuples)}, anchors without positives: {n_no_pos}")

    with open(OUT / "train_nsc.pickle", "wb") as f:
        pickle.dump(tuples, f)
    print(f"wrote {OUT/'train_nsc.pickle'} "
          f"({(OUT/'train_nsc.pickle').stat().st_size/1e6:.1f} MB)")

    # ---- tiny val pickle (constructor fodder only, never iterated) ----
    val_src = ("kitti", "02")
    s = seqs[val_src]
    n_val = min(400, len(s["scan_ids"]))
    xy = s["poses"][:n_val, :2, 3]
    tree = cKDTree(xy)
    vt = {}
    for i in range(n_val):
        pos = np.sort(np.asarray(tree.query_ball_point(xy[i], POS_M), dtype=np.int64))
        nn = np.sort(np.asarray(tree.query_ball_point(xy[i], NONNEG_M), dtype=np.int64))
        vt[i] = TrainingTuple(
            id=i, timestamp=int(s["ts"][i]),
            rel_scan_filepath=f"velodyne_sync/{val_src[0]}/{val_src[1]}/{int(s['scan_ids'][i]):07d}.pc",
            positives=pos[pos != i].astype(np.int32),
            non_negatives=nn.astype(np.int32),
            pose=s["poses"][i], positives_poses={})
    with open(OUT / "val_nsc.pickle", "wb") as f:
        pickle.dump(vt, f)
    print(f"wrote {OUT/'val_nsc.pickle'} ({n_val} tuples from {val_src})")

    manifest = {
        "split": "pd2026 place-disjoint (configs/training_place_disjoint.yaml)",
        "train_sequences": [f"{d}/{s}" for d, s in train_list],
        "n_train_tuples": len(tuples),
        "pos_m": POS_M, "nonneg_m": NONNEG_M,
        "grouping": "nclt sessions merged (co-registered); others per-sequence",
        "virtual_path": "velodyne_sync/{dtype}/{seq}/{scan_id}.pc",
        "cache_keys": CACHE_KEYS,
        "checkpoint_selection_rule": (
            "highest pd-validation sequence-macro 5 m Recall@1 over epochs 1-5; "
            "earlier epoch on ties; weight sha256 frozen before any test contact"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written")


if __name__ == "__main__":
    main()
