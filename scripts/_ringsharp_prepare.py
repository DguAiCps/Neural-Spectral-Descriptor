#!/usr/bin/env python3
"""Prepare RINGSharp-format training pickles from OUR 24-sequence split.

Builds, under _external/ringsharp_data/ (the "staging dir"):
  * train_nsc.pickle        — dict ndx -> glnet TrainingTuple over all 24 train
                              sequences (global consecutive ids, virtual scan
                              paths 'velodyne_sync/{dtype}/{seq}/{scan_id}.pc')
  * val_nsc.pickle          — small TrainingTuple pickle (subset of kitti 02);
                              only needed so make_dataloaders(validation=True)
                              constructs; never iterated (train runs w/o --val)
  * val_nsc_evalset.pickle  — glnet EvaluationSet over mulran DCC03 (map =
                              first-visit keyframes >=20 m apart, query =
                              revisit keyframes, our 5 m / 30-frame revisit
                              rule) — only feeds RINGSharp's own side-channel
                              sanity evaluator at the end of training
  * manifest.json           — provenance

Positive/non-negative semantics follow the official RINGSharp NCLT recipe
(glnet/datasets/nclt/generate_training_tuples.py): positives = same-group
frames within 25 m (anchor removed), non_negatives = same-group frames within
50 m (anchor kept). positives_poses is left empty — with the RING#-L LiDAR
config the trainer computes relative poses from the absolute poses in-batch
(YawLoss / TransLoss), and ICP_REFINE is off upstream as well.

Grouping (documented deviation): NCLT train sessions are co-registered in one
campus frame (median cross-session NN distance 0.5-1.2 m in our caches), so
the five NCLT sequences form ONE group with cross-session positives, matching
the official multi-session NCLTSequences behavior. KITTI / HeLiPR / MulRan
sequences are per-sequence groups: our cached MulRan pose frames are
origin-normalized per sequence (median cross-sequence NN offset 39-240 m), so
cross-sequence geometry is unavailable; same-region MulRan sequences may
therefore produce rare false in-batch negatives (<~ a few % of batches) —
same treatment as the SeqOT baseline adaptation.

Run inside docker (no GPU needed):
  python scripts/_ringsharp_prepare.py
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

from glnet.datasets.base_datasets import TrainingTuple, EvaluationSet, EvaluationTuple  # noqa: E402

OUT = REPO / "_external" / "ringsharp_data"
CONFIG = REPO / "configs" / "training_multi_dataset.yaml"

CACHE_KEYS = {"kitti": "056e0a02", "nclt": "056e0a02",
              "helipr": "056e0a02", "mulran": "33919e6e"}
# Same fallbacks as scripts/_seqot_prepare.py
CACHE_KEY_FALLBACK = {("kitti", "01"): "de201724",
                      ("nclt", "2012-11-16"): "76c42cd7",
                      ("nclt", "2013-02-23"): "76c42cd7"}

POS_M, NONNEG_M = 25.0, 50.0            # official RINGSharp NCLT thresholds
REVISIT_M, SKIP_FRAMES = 5.0, 30        # our protocol (for the sanity evalset)
MAP_SAMPLING_M = 20.0                   # their protocol map sampling


def cache_path(dtype, seq, split):
    key = CACHE_KEYS[dtype]
    name = f"cache_{key}_{dtype}_val_{seq}.npz" if split == "val" \
        else f"cache_{key}_{dtype}_{seq}.npz"
    p = REPO / "data" / "preprocessed" / name
    if not p.exists() and split != "val" and (dtype, str(seq)) in CACHE_KEY_FALLBACK:
        fb = CACHE_KEY_FALLBACK[(dtype, str(seq))]
        p = REPO / "data" / "preprocessed" / f"cache_{fb}_{dtype}_{seq}.npz"
    return p


# KITTI caches store cam0-convention poses (x-right, y-down, z-forward; the
# cached translation ranges are x 598 / y 58 / z 946 m on seq 02). The RING#
# losses interpret pose (x, y) as the ground plane and yaw as rotation about
# z, so KITTI poses are conjugated into a z-up body frame
# (x-forward, y-left, z-up): P' = M P M^-1 with M the axis permutation below.
# The small cam0<->velodyne extrinsic offset (~0.1 m) is ignored — irrelevant
# at the 0.875 m BEV grid resolution.
_R_BC = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
_M_BC = np.eye(4)
_M_BC[:3, :3] = _R_BC


def fix_pose_frame(dtype, poses):
    if dtype != "kitti":
        return poses
    return np.einsum("ij,njk,kl->nil", _M_BC, poses, np.linalg.inv(_M_BC))


def load_seq(dtype, seq, split):
    with np.load(cache_path(dtype, seq, split), allow_pickle=True) as c:
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
    assert len(train_list) == 24, f"expected 24 train sequences, got {len(train_list)}"

    # ---- load all train sequences ----
    seqs = {}
    for dtype, seq in train_list:
        sids, poses, ts = load_seq(dtype, seq, "train")
        seqs[(dtype, seq)] = dict(scan_ids=sids, poses=poses, ts=ts)
        print(f"[{dtype}/{seq}] {len(sids)} keyframes")

    # ---- grouping: one group for all NCLT sessions, per-sequence otherwise ----
    def group_of(dtype, seq):
        return "nclt_campus" if dtype == "nclt" else f"{dtype}_{seq}"

    # global ids
    offset = 0
    order = []
    for key in train_list:
        n = len(seqs[key]["scan_ids"])
        seqs[key]["gid0"] = offset
        order.append(key)
        offset += n
    n_total = offset
    print(f"total {n_total} train keyframes")

    # group -> (global ids array, positions array)
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

    # ---- build TrainingTuples ----
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

    # ---- tiny val pickle (constructor fodder only) ----
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

    # ---- sanity EvaluationSet over mulran DCC03 (their evaluator format) ----
    sids, poses, ts = load_seq("mulran", "DCC03", "val")
    xy = poses[:, :2, 3]
    tree = cKDTree(xy)
    # queries: our revisit rule (first earlier frame <5 m with >=30-frame gap)
    q_idx = []
    for j in range(SKIP_FRAMES, len(xy)):
        for i in tree.query_ball_point(xy[j], REVISIT_M):
            if i <= j - SKIP_FRAMES:
                q_idx.append(j)
                break
    q_set = set(q_idx)
    # map: first-visit keyframes, greedily sampled at >=20 m spacing
    map_idx, kept = [], []
    for i in range(len(xy)):
        if i in q_set:
            continue
        if not kept or np.min(np.linalg.norm(np.array(kept) - xy[i], axis=1)) >= MAP_SAMPLING_M:
            map_idx.append(i)
            kept.append(xy[i])

    def ev_tuple(i):
        return EvaluationTuple(
            timestamp=int(ts[i]),
            rel_scan_filepath=f"velodyne_sync/mulran/DCC03/{int(sids[i]):07d}.pc",
            position=xy[i], pose=poses[i])

    es = EvaluationSet([ev_tuple(i) for i in q_idx], [ev_tuple(i) for i in map_idx])
    es.save(str(OUT / "val_nsc_evalset.pickle"))
    print(f"wrote {OUT/'val_nsc_evalset.pickle'} "
          f"(map {len(map_idx)}, query {len(q_idx)})")

    manifest = {
        "train_sequences": [f"{d}/{s}" for d, s in train_list],
        "n_train_tuples": len(tuples),
        "pos_m": POS_M, "nonneg_m": NONNEG_M,
        "grouping": "nclt sessions merged (co-registered); others per-sequence",
        "virtual_path": "velodyne_sync/{dtype}/{seq}/{scan_id}.pc",
        "cache_keys": CACHE_KEYS, "cache_key_fallback":
            {f"{k[0]}/{k[1]}": v for k, v in CACHE_KEY_FALLBACK.items()},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written")


if __name__ == "__main__":
    main()
