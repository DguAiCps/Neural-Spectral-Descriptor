#!/usr/bin/env python3
"""HeLiOS official pipeline on split v2 -- part 2: overlap labels and the
official training / validation pickles (2026-10-11).

Official definition (all unchanged in meaning, see `selftest`):
  _external/HeLiOS/datasets/preprocess/LiDAR_transformer.py
      apply_transformation(pc, [x, y, z, qx, qy, qz, qw]): the toolbox scan
      (normalised by 1/100) is moved into the shared frame with the LiDAR
      pose, translation x 0.01; stored as float32 (save_pc).  Executed from
      the original file (function extracted with ast; the module body would
      iterate over hard-coded folders).
  _external/HeLiOS/datasets/preprocess/calculate_overlap/src/pointcloud_listener.cpp
      for every scan pair: 0 if the trajectory.csv (northing, easting) are
      more than 200 m apart (Eigen::Vector3f); else PCL VoxelGrid 0.04 on both
      clouds (= 4 m), overlap(a, b) = 2 * #{a points whose nearest b point is
      closer than 0.06 (= 6 m)} / (|a| + |b|), capped at 1, max of both
      directions; written as text with 5 significant digits.
  _external/HeLiOS/datasets/preprocess/generate_training_sets.py
      construct_query_dict: positives = overlap > 0.5 (query removed),
      non_negatives (semi-positives) = 0 < overlap <= 0.5, everything else is a
      negative (this includes the query itself: the official collate_fn
      therefore marks the diagonal as negative -- reproduced, not changed).
      One matrix over ALL folders of the role (pairs across sequences and
      sensors are labelled whenever the poses are within 200 m).
Re-implemented here for scale (the official binary keeps a dense N x N double
matrix and a dense text file): the same arithmetic on sparse pairs --
PCL 1.14 VoxelGrid semantics in float32 (cell = floor(p * (1/0.04f)) -
min_b; centroid = float32 running sum / n in a stable cell order; PCL itself
sorts cells with an unstable sort, so the sum order inside a cell, i.e. the
last float bit of a centroid, is library-version dependent), nearest
neighbour exact (scipy cKDTree) with the FLANN float32 squared distance
((dx*dx + dy*dy) + dz*dz < 0.06^2), the official 5-significant-digit rounding
before thresholding.  `selftest` runs the official binary and the official
construct_query_dict on real subsets and compares.

What is ours (protocol / data):
  * scans: split-v2 keyframes after the official toolbox preprocessing
    (scripts/_helios_official_prep.py), TRAIN keyframes thinned by the
    official 3 m rule (index --train-spacing), validation = Roundabout01
    (official validation_set, a split-v2 VAL sequence), 3 m rule;
  * poses: cache LiDAR poses (HeLiPR: LiDAR_GT/Velodyne_gt.txt rows; KITTI:
    camera-0 pose @ calib Tr).  Only HeLiPR Bridge01-04 share a frame in our
    caches; each KITTI sequence has its own local frame, so pairs are formed
    inside a frame group only (official: one global frame).  KITTI 05/06 come
    within ~50 m of each other in the georeferenced audit frame; those pairs
    are negatives here (disclosed);
  * (northing, easting) = horizontal axes of the LiDAR position in the group
    frame (KITTI camera world frame: x, z).

Subcommands (inside ring_baseline):
  vox --role train|val            transformed + voxelised clouds per scan
  pairs --role R [--shard k --nshards n]   overlaps of pairs (resumable)
  pickle --role R                 official pickle -> data/{training,validation}.pickle
  selftest                        replica == official binary + generator (CPU)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

import _helios_official_prep as P  # noqa: E402

PRE = P.HELIOS / "datasets/preprocess"
OVL = P.WORK / "overlap"
LEAF = 0.04            # pointcloud_listener.cpp:226-231
DIS_THRESHOLD = 0.06   # pointcloud_listener.cpp:233
GATE_M = 200.0         # pointcloud_listener.cpp:213
OVERLAP_THRESHOLD = 0.5  # generate_training_sets.py:50
PICKLE = {"train": "training.pickle", "val": "validation.pickle"}


def _official_functions(path: Path, names):
    """Execute only the named top-level functions of an official file."""
    src = path.read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.FunctionDef) and n.name in names)]
    ns = {"__file__": str(path)}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return [ns[n] for n in names]


def official_transformer():
    return _official_functions(PRE / "LiDAR_transformer.py", ["apply_transformation", "load_pc"])


def official_construct_query_dict():
    return _official_functions(PRE / "generate_training_sets.py", ["construct_query_dict"])[0]


# ------------------------------------------------------------ membership
def role_members(role: str):
    """[(folder, sensor, seq, ids-in-index, scan_ids, lidar_poses, timestamps)]
    in the official order: folders sorted by name (generate_training_sets.py:52
    sorted(os.listdir(...)); the overlap binary sorts the same way), rows in
    trajectory (= scan) order."""
    if role == "train":
        seqs = P.ROLES["train"][0]
    elif role == "val":
        seqs = P.OFFICIAL_VAL
    else:
        raise ValueError(role)
    out = []
    for sensor, seq in seqs:
        z = np.load(P.seq_dir(role, sensor, seq) / "index.npz")
        keep = np.flatnonzero(z["keep"])
        out.append((P.seq_name(sensor, seq), sensor, seq, keep, z["scan_ids"][keep],
                    z["lidar_poses"][keep], z["timestamps"][keep]))
    return sorted(out, key=lambda m: m[0])


def pose_to_transformation(T: np.ndarray) -> np.ndarray:
    """4x4 -> [x, y, z, qx, qy, qz, qw] (trajectory_transform.csv row order)."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(T[:3, :3]).as_quat()   # x, y, z, w
    return np.concatenate([T[:3, 3], q])


# ------------------------------------------------------------ overlap arithmetic
def pcl_voxel_grid(p: np.ndarray, leaf: float = LEAF) -> np.ndarray:
    """pcl::VoxelGrid<PointXYZ> (PCL 1.14 voxel_grid.hpp applyFilter,
    downsample_all_data, min_points_per_voxel 0) in float32."""
    p = np.ascontiguousarray(p, dtype=np.float32)
    inv = np.float32(1.0) / np.float32(leaf)
    mn, mx = p.min(0), p.max(0)
    d = ((mx - mn) * inv).astype(np.int64) + 1
    if int(d[0]) * int(d[1]) * int(d[2]) > np.iinfo(np.int32).max:
        return p.copy()                      # PCL: leaf too small -> input unchanged
    min_b = np.floor(mn * inv).astype(np.int64)
    max_b = np.floor(mx * inv).astype(np.int64)
    div = max_b - min_b + 1
    ijk = (np.floor(p * inv) - min_b.astype(np.float32)).astype(np.int64)
    idx = ijk[:, 0] + ijk[:, 1] * div[0] + ijk[:, 2] * div[0] * div[1]
    order = np.argsort(idx, kind="stable")
    sidx = idx[order]
    first = np.flatnonzero(np.r_[True, sidx[1:] != sidx[:-1]])
    cell = np.cumsum(np.r_[0, (sidx[1:] != sidx[:-1]).astype(np.int64)])
    sums = np.zeros((len(first), 3), np.float32)
    np.add.at(sums, cell, p[order])          # sequential float32 running sums
    cnt = np.diff(np.r_[first, len(sidx)]).astype(np.float32)
    return sums / cnt[:, None]


def _matches(a: np.ndarray, b: np.ndarray, tree_b) -> int:
    """#{a points whose nearest b point has float32 squared distance < 0.06^2}
    (KdTreeFLANN nearestKSearch, L2_Simple<float>)."""
    _, nn = tree_b.query(a.astype(np.float64), k=1)
    dd = a - b[nn]
    s = (dd[:, 0] * dd[:, 0] + dd[:, 1] * dd[:, 1]) + dd[:, 2] * dd[:, 2]
    return int(np.count_nonzero(s.astype(np.float64) < DIS_THRESHOLD * DIS_THRESHOLD))


def symmetric_overlap(a, b, tree_a, tree_b) -> float:
    n = len(a) + len(b)
    o1 = min(1.0, 2.0 * _matches(a, b, tree_b) / n)
    o2 = min(1.0, 2.0 * _matches(b, a, tree_a) / n)
    return max(o1, o2)


def text5(x: float) -> float:
    """std::ostream << double with precision(5), parsed back (np.fromstring)."""
    return float(f"{x:.5g}")


def gate_far(hi: np.ndarray, hj: np.ndarray) -> np.ndarray:
    """(Vector3f(n_i, e_i, 0) - Vector3f(n_j, e_j, 0)).norm() > 200, float32."""
    d = hi.astype(np.float32) - hj.astype(np.float32)
    return np.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]) > np.float32(GATE_M)


# ------------------------------------------------------------ stages
def cmd_vox(role: str):
    apply_transformation, load_pc = official_transformer()
    members = role_members(role)
    pts, offs, groups, horiz, rel, folder_of, stamps = [], [0], [], [], [], [], []
    t0 = time.time()
    for folder, sensor, seq, _, sids, lps, ts in members:
        for sid, T, t in zip(sids, lps, ts):
            pc = load_pc(str(P.DATA / P.rel_scan_path(role, sensor, seq, sid)))
            tr = apply_transformation(pc, pose_to_transformation(T)).astype(np.float32)
            v = pcl_voxel_grid(tr)
            pts.append(v)
            offs.append(offs[-1] + len(v))
            rel.append(P.rel_scan_path(role, sensor, seq, sid))
            stamps.append(int(round(float(t) * 1e9)))
        groups += [P.frame_group(sensor, seq)] * len(sids)
        horiz.append(P.horizontal(sensor, lps[:, :3, 3]))
        folder_of += [folder] * len(sids)
        print(f"  vox {folder}: {len(sids)} scans ({time.time() - t0:.0f} s)", flush=True)
    out = OVL / role
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "vox.npz", points=np.concatenate(pts), offsets=np.asarray(offs, np.int64),
             group=np.asarray(groups), horizontal=np.concatenate(horiz),
             rel=np.asarray(rel), folder=np.asarray(folder_of), timestamp=np.asarray(stamps))
    nv = np.diff(offs)
    print(f"OK vox {role}: {len(nv)} scans, voxels/scan median {np.median(nv):.0f} "
          f"(min {nv.min()}, max {nv.max()})", flush=True)


def _candidates(i, H, G, n):
    j = np.arange(i + 1, n)
    j = j[G[j] == G[i]]
    return j[~gate_far(H[i][None], H[j])]


def cmd_pairs(role: str, shard: int, nshards: int):
    from scipy.spatial import cKDTree
    out = OVL / role / f"pairs_{shard:03d}_of_{nshards:03d}.npz"
    if out.exists():
        print(f"[skip] {out}", flush=True)
        return
    z = np.load(OVL / role / "vox.npz")
    pts, offs, G, H = z["points"], z["offsets"], z["group"], z["horizontal"]
    n = len(offs) - 1
    cloud = lambda k: pts[offs[k]:offs[k + 1]]  # noqa: E731
    trees = {}

    def tree(k):
        t = trees.get(k)
        if t is None:
            if len(trees) > 6000:
                trees.clear()
            t = trees[k] = cKDTree(cloud(k).astype(np.float64))
        return t

    I, J, O = [], [], []
    n_pairs, t0 = 0, time.time()
    mine = range(shard, n, nshards)
    for c, i in enumerate(mine):
        a, ta = cloud(i), tree(i)
        for j in _candidates(i, H, G, n):
            ov = text5(symmetric_overlap(a, cloud(j), ta, tree(j)))
            n_pairs += 1
            if ov > 0:
                I.append(i)
                J.append(j)
                O.append(ov)
        if (c + 1) % 200 == 0:
            print(f"  [{role} {shard}/{nshards}] {c + 1}/{len(mine)} queries, {n_pairs} pairs "
                  f"({(time.time() - t0) / max(1, n_pairs) * 1e3:.3f} ms/pair)", flush=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, i=np.asarray(I, np.int32), j=np.asarray(J, np.int32),
             overlap=np.asarray(O, np.float64), n_pairs=n_pairs)
    os.replace(tmp, out)
    print(f"OK pairs {role} {shard}/{nshards}: {n_pairs} pairs, {len(O)} with overlap > 0, "
          f"{time.time() - t0:.0f} s", flush=True)


def build_queries(n, I, J, O, rel, stamps, H):
    """construct_query_dict on the sparse symmetric matrix (diagonal 1.0)."""
    ii = np.concatenate([I, J]).astype(np.int64)
    jj = np.concatenate([J, I]).astype(np.int64)
    oo = np.concatenate([O, O])
    order = np.lexsort((jj, ii))
    ii, jj, oo = ii[order], jj[order], oo[order]
    start = np.searchsorted(ii, np.arange(n + 1))
    queries = {}
    for q in range(n):
        js, os_ = jj[start[q]:start[q + 1]], oo[start[q]:start[q + 1]]
        pos = js[os_ > OVERLAP_THRESHOLD]
        semi = js[(os_ > 0) & (os_ <= OVERLAP_THRESHOLD)]
        queries[q] = {"id": q, "timestamp": int(stamps[q]), "rel_scan_filepath": str(rel[q]),
                      "positives": pos, "non_negatives": semi,
                      "position": np.array([H[q][0], H[q][1]], dtype=np.float64),
                      "pos_overlap": [os_[os_ > OVERLAP_THRESHOLD]],
                      "semi_pos_overlap": [os_[(os_ > 0) & (os_ <= OVERLAP_THRESHOLD)]]}
    return queries


def cmd_pickle(role: str, out_path: Path | None = None):
    z = np.load(OVL / role / "vox.npz")
    n = len(z["offsets"]) - 1
    shards = sorted((OVL / role).glob("pairs_*_of_*.npz"))
    shards = [s for s in shards if ".tmp" not in s.name]
    nsh = {int(s.stem.split("_of_")[1]) for s in shards}
    assert len(nsh) == 1 and len(shards) == nsh.pop(), f"incomplete shards: {[s.name for s in shards]}"
    parts = [np.load(s) for s in shards]
    I = np.concatenate([p["i"] for p in parts])
    J = np.concatenate([p["j"] for p in parts])
    O = np.concatenate([p["overlap"] for p in parts])
    n_pairs = int(sum(int(p["n_pairs"]) for p in parts))
    q = build_queries(n, I, J, O, z["rel"], z["timestamp"], z["horizontal"])
    out_path = out_path or (P.DATA / PICKLE[role])
    with open(out_path, "wb") as f:
        pickle.dump(q, f, protocol=pickle.HIGHEST_PROTOCOL)
    npos = np.array([len(v["positives"]) for v in q.values()])
    nsemi = np.array([len(v["non_negatives"]) for v in q.values()])
    stats = {"role": role, "pickle": str(out_path.relative_to(REPO)), "n_queries": n,
             "n_pairs_within_200m": n_pairs, "n_pairs_overlap_gt0": int(len(O)),
             "positives_per_query": {"median": float(np.median(npos)), "min": int(npos.min()),
                                     "max": int(npos.max()), "n_zero": int((npos == 0).sum())},
             "semi_positives_per_query": {"median": float(np.median(nsemi)),
                                          "max": int(nsemi.max())},
             "cross_sequence_positive_pairs": int(np.sum(
                 (z["folder"][I] != z["folder"][J]) & (O > OVERLAP_THRESHOLD))),
             "pickle_bytes": out_path.stat().st_size,
             "frame_groups": sorted(set(z["group"].tolist()))}
    (OVL / role / "pickle_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


# ------------------------------------------------------------ selftest
def _official_overlap_matrix(work: Path, folders):
    """Run the official overlap binary on <work>/runs/<folder>/{trajectory.csv,
    LiDAR_transformed/*.bin}; returns the matrix parsed like the generator."""
    import yaml
    (work / "config").mkdir(parents=True, exist_ok=True)
    (work / "build").mkdir(exist_ok=True)
    cfg = {"base_path": str(work) + "/", "runs_folder": "runs/", "filename": "trajectory.csv",
           "dir_txt": "overlap_matrix.txt", "folder_list": list(folders)}
    (work / "config/config.yaml").write_text(yaml.safe_dump(cfg))
    r = subprocess.run([str(P.OVERLAP_BIN)], cwd=work / "build", capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rows = [np.fromstring(line, dtype=float, sep=" ")
            for line in (work / "overlap_matrix.txt").read_text().splitlines()]
    return np.vstack(rows)


def selftest(n_kitti: int, n_bridge: int, out_json: Path | None):
    """Two official runs: KITTI 06 (own frame) and Bridge01 + Bridge04 (shared
    frame, cross-sequence pairs).  Scans: official toolbox preprocessing of the
    first kept TRAIN keyframes (3 m rule).  Compares (1) every matrix entry of
    the official binary with our replica, (2) the official construct_query_dict
    pickle on the official matrix with ours on the sparse pairs."""
    import pandas as pd
    from scipy.spatial import cKDTree
    from _pd_build_test_caches import _make_loader
    apply_transformation, load_pc = official_transformer()
    construct_query_dict = official_construct_query_dict()
    tb = P.Toolbox()
    rep = {"overlap_binary_sha256": P.sha256(P.OVERLAP_BIN), "runs": []}
    ok_all = True
    with tempfile.TemporaryDirectory(dir=str(P.WORK)) as td:
        td = Path(td)
        for run, seqs, nsc in (("kitti06", [("kitti", "06")], n_kitti),
                               ("bridge01_04", [("helipr", "Bridge01"), ("helipr", "Bridge04")], n_bridge)):
            w = td / run
            members, H, groups, rel, stamps = [], [], [], [], []
            vox = []
            for sensor, seq in seqs:
                z = np.load(P.cache_path("train", sensor, seq))
                poses = z["poses"].astype(np.float64)
                sids_all = z["scan_ids"]
                lp = P.lidar_poses(sensor, seq, poses)
                keep = np.flatnonzero(P.official_keyframes(lp[:, :3, 3], P.TOOLBOX_SPACING_M))[:nsc]
                folder = P.seq_name(sensor, seq)
                d = w / "runs" / folder
                (d / "LiDAR_transformed").mkdir(parents=True)
                loader = _make_loader(sensor, seq)
                rows = []
                for k in keep:
                    sid = int(sids_all[k])
                    pts, _ = P.process_scan(tb, loader[sid]["points"], P.fill_rng(sensor, seq, sid))
                    raw = w / "toolbox" / folder / f"{sid:06d}.bin"
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    pts.astype("<f4").tofile(raw)
                    pc = load_pc(str(raw))                       # official
                    tr = apply_transformation(pc, pose_to_transformation(lp[k]))
                    tr.astype(np.float32).tofile(d / "LiDAR_transformed" / f"{sid:06d}.bin")  # save_pc
                    h = P.horizontal(sensor, lp[k][None, :3, 3])[0]
                    rows.append((f"{sid:06d}", h[0], h[1]))
                    vox.append(pcl_voxel_grid(tr.astype(np.float32)))
                    H.append(h)
                    groups.append(P.frame_group(sensor, seq))
                    rel.append(f"runs/{folder}/LiDAR/{sid:06d}.bin")
                    stamps.append(sid)
                pd.DataFrame(rows, columns=["timestamp", "northing", "easting"]).to_csv(
                    d / "trajectory.csv", index=False)
                members.append(folder)
            t0 = time.time()
            M = _official_overlap_matrix(w, sorted(members))
            t_off = time.time() - t0
            # ours
            H, G = np.asarray(H), np.asarray(groups)
            n = len(vox)
            trees = [cKDTree(v.astype(np.float64)) for v in vox]
            ours = np.zeros((n, n))
            np.fill_diagonal(ours, 1.0)
            I, J, O = [], [], []
            t0 = time.time()
            for i in range(n):
                for j in _candidates(i, H, G, n):
                    ov = text5(symmetric_overlap(vox[i], vox[j], trees[i], trees[j]))
                    ours[i, j] = ours[j, i] = ov
                    if ov > 0:
                        I.append(i), J.append(j), O.append(ov)
            t_ours = time.time() - t0
            diff = np.abs(M - ours)
            n_diff = int((diff > 0).sum())
            diag = []
            for i, j in zip(*np.nonzero(np.triu(diff > 0))):
                # why: distance of the nearest neighbours to the 0.06 threshold
                # and the overlap with float64 centroid sums (sum-order sensitivity)
                near = []
                for a, b, tb_ in ((vox[i], vox[j], trees[j]), (vox[j], vox[i], trees[i])):
                    dist, _ = tb_.query(a.astype(np.float64), k=1)
                    near.append(float(np.min(np.abs(dist - DIS_THRESHOLD))))
                diag.append({"i": int(i), "j": int(j), "official": float(M[i, j]), "ours": float(ours[i, j]),
                             "n_vox": [int(len(vox[i])), int(len(vox[j]))],
                             "closest_nn_distance_to_threshold": min(near)})

            cls = lambda m: (m > OVERLAP_THRESHOLD).astype(int) + (m > 0)  # noqa: E731
            n_cls_diff = int((cls(M) != cls(ours)).sum())
            # official generator on the official matrix vs our sparse builder
            df = pd.DataFrame({"file": rel, "northing": H[:, 0], "easting": H[:, 1]})
            off_pickle = w / "official.pickle"
            construct_query_dict(df, str(off_pickle), str(w / "overlap_matrix.txt"), OVERLAP_THRESHOLD)
            with open(off_pickle, "rb") as f:
                off_q = pickle.load(f)
            our_q = build_queries(n, np.asarray(I), np.asarray(J), np.asarray(O),
                                  np.asarray(rel), np.asarray(stamps), H)
            same_q, n_val_diff = True, 0
            for q in range(n):
                a, b = off_q[q], our_q[q]
                for k in ("positives", "non_negatives"):
                    same_q &= np.array_equal(np.asarray(a[k], np.int64), b[k])
                for k in ("pos_overlap", "semi_pos_overlap"):
                    same_q &= len(a[k][0]) == len(b[k][0])
                    n_val_diff += int(np.sum(a[k][0] != b[k][0])) if len(a[k][0]) == len(b[k][0]) else 0
                same_q &= a["rel_scan_filepath"] == b["rel_scan_filepath"] and a["id"] == b["id"]
                same_q &= a["timestamp"] == b["timestamp"] and np.allclose(a["position"], b["position"])
            npos = [len(off_q[q]["positives"]) for q in range(n)]
            r = {"run": run, "n_scans": n, "n_entries": n * n, "n_entries_differing": n_diff,
                 "max_abs_diff": float(diff.max()), "n_class_flips": n_cls_diff,
                 "pickle_labels_equal": bool(same_q),
                 "pickle_overlap_values_differing": n_val_diff,
                 "differing_pairs": diag, "differences_boundary_only": None,
                 "positives_per_query_median": float(np.median(npos)),
                 "official_binary_s": round(t_off, 1), "replica_s": round(t_ours, 1),
                 "pairs_computed": int(sum(len(_candidates(i, H, G, n)) for i in range(n)))}
            # accepted: identical labels (positive / semi-positive / negative sets);
            # a value may differ only where a nearest-neighbour distance sits within
            # 1e-6 of the 0.06 threshold (last float bit of a centroid) and by at
            # most two matched points (+ the 5-digit rounding)
            boundary_only = all(d["closest_nn_distance_to_threshold"] < 1e-6 and
                                abs(d["official"] - d["ours"]) <= 4.0 / sum(d["n_vox"]) + 1e-4
                                for d in diag)
            ok_all &= n_cls_diff == 0 and same_q and boundary_only and n_diff <= 1e-3 * n * n
            r["differences_boundary_only"] = bool(boundary_only)
            rep["runs"].append(r)
            print(json.dumps(r), flush=True)
    rep["all_ok"] = bool(ok_all)
    if out_json:
        out_json.write_text(json.dumps(rep, indent=1))
    print("OVERLAP_SELFTEST_OK" if ok_all else "OVERLAP_SELFTEST_FAILED", flush=True)
    if not ok_all:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("vox")
    p.add_argument("--role", choices=("train", "val"), required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--role", choices=("train", "val"), required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p = sub.add_parser("pickle")
    p.add_argument("--role", choices=("train", "val"), required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--n-kitti", type=int, default=120)
    p.add_argument("--n-bridge", type=int, default=90)
    p.add_argument("--out-json", type=Path, default=P.LOG_DIR / "helios_selftest_overlap.json")
    a = ap.parse_args()
    if a.cmd == "vox":
        cmd_vox(a.role)
    elif a.cmd == "pairs":
        cmd_pairs(a.role, a.shard, a.nshards)
    elif a.cmd == "pickle":
        cmd_pickle(a.role)
    elif a.cmd == "selftest":
        selftest(a.n_kitti, a.n_bridge, a.out_json)


if __name__ == "__main__":
    main()
