#!/usr/bin/env python3
"""BEVPlace++ official pipeline on split v2 -- part 1: BEV images and the
shared official-code / paper-override helpers (2026-10-11).

Feeds our loaders' points into the OFFICIAL generator of
_external/BEVPlace2 (zjuluolun/BEVPlace2, commit OFFICIAL_COMMIT):
  * datasets/gen_bev_images.py `getBEV` is executed from the unmodified file
    (0.4 m open3d voxel down-sample, point count per 0.4 m cell saturated at
    10, x10, /max*255, row = 100 - floor(x/0.4), col = 100 - floor(y/0.4));
  * the three |x|,|y|,|z| < 40 m filters and the float32 cast of that file's
    __main__ block (lines 66-76) are repeated here verbatim (a __main__ block
    cannot be imported);
  * the image is written with cv2.imwrite exactly as line 79 does (float64
    image -> OpenCV's own CV_8U conversion).
PAPER OVERRIDE (project rule 2026-10-11: paper value wins over code):
  * image size 200x200 (paper Sec. V implementation details: D = 40 m,
    g = 0.4 m, "the BEV image has a size of 200x200", ceil(2D/g)) instead of
    the code's 201x201 (gen_bev_images.py:28-36).  Row 0 and column 0 of the
    code's image are never hit (|x|,|y| < 40 m -> index 1..200), so the
    paper image is the code image without them; asserted on every image.
The only inputs that are ours: which scans (the keyframe scan_ids of the
split-v2 caches) and their points (scripts/_pd_build_test_caches._make_loader,
sensor frame, z up; NCLT already flipped by the loader).

Layout = the official data layout (datasets/data.md), one root per role:
  _external/bevplace_official_pdv2/bev/{role}/{sensor}_{seq}/bev_imgs/{scan_id:06d}.png
  _external/bevplace_official_pdv2/bev/{role}/poses/{sensor}_{seq}.txt   (N x 12)
  _external/bevplace_official_pdv2/bev/{role}/{sensor}_{seq}/{scan_ids.npy,meta.json}
  _external/bevplace_official_pdv2/points/train/{sensor}_{seq}/{scan_id:06d}.npy
      (train only: float32 xyz with xy-radius < 56.6 m and |z| < 40 m -- the
       only points that can enter the +/-40 m window after any z-rotation;
       input of the paper's point-cloud rotation augmentation)
roles: train (v2 TRAIN, GT keyframe caches), val (v2 VAL, GT keyframe
caches), test_odom (v2 TEST, KISS-ICP keyframe caches; the poses file holds
poses_gt and is used for scoring only -- the method never reads poses at test).

Subcommands (run inside ring_baseline):
  index --role R            poses txt + scan_ids + meta for every sequence
  jobs  [--chunk 2000]      print "role seq shard nshards" generation jobs
  gen --role R --seq S [--shard k --nshards n]   write the PNGs (resumable)
  status                    count PNGs (and train point files) against the caches
  selftest [--n-scans 2]    CPU: adapter output == official script output;
                            fast getBEV == official getBEV under rotations
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from _pdv2_pilot import TEST, TRAIN, VAL  # noqa: E402  (split-v2 lists)

OFFICIAL = REPO / "_external/BEVPlace2"
OFFICIAL_COMMIT = "2d0636d59fac6520bead4faf3f933be592bcd635"
PYLIB = REPO / "_external/bevplace_pylib"     # h5py 3.16.0 (pip --target)
WORK = REPO / "_external/bevplace_official_pdv2"
BEV = WORK / "bev"
POINTS = WORK / "points"
IMG = 200        # paper: ceil(2D/g) = 200 (code: 201)
R_PRE = 56.6     # > 40*sqrt(2) = 56.57 m: farther points never enter the window
V2 = REPO / "data/preprocessed_v2"
ODOM = REPO / "data/preprocessed_v2_odom"
# role -> (sequence list, cache dir, cache-name middle, pose key in cache)
ROLES = {
    "train": (TRAIN, V2, "", "poses"),
    "val": (VAL, V2, "val_", "poses"),
    "test_odom": (TEST, ODOM, "test_", "poses_gt"),
}


def seq_name(sensor: str, seq: str) -> str:
    return f"{sensor}_{seq}"


def cache_path(role: str, sensor: str, seq: str) -> Path:
    _, cdir, mid, _ = ROLES[role]
    hits = sorted(glob.glob(str(cdir / f"cache_????????_{sensor}_{mid}{seq}.npz")))
    assert len(hits) == 1, (role, sensor, seq, hits)
    return Path(hits[0])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------ official code
def official_getBEV():
    """`getBEV` from the unmodified datasets/gen_bev_images.py. The file ends
    with a module-level exit(); it is executed as a non-__main__ module and
    the SystemExit raised by that last line is caught after getBEV exists."""
    spec = importlib.util.spec_from_file_location(
        "bevplace2_gen_bev_images", OFFICIAL / "datasets/gen_bev_images.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod.getBEV


def load_official():
    """Import the official training/eval modules from _external/BEVPlace2.

    Environment shims only (no code of the official files is changed):
      * h5py is not installed in ring_baseline -> pip --target copy in PYLIB;
      * `imgaug` is imported but never used by kitti_dataset.py /
        nclt_dataset.py and is not installed -> empty stub module.
    main.py is loaded under the name `bevplace2_main` (its functions read
    module globals opt / device / model / optimizer / writer, which the
    drivers set on the returned module)."""
    for p in (str(PYLIB), str(OFFICIAL)):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    if "imgaug" not in sys.modules:
        stub = types.ModuleType("imgaug")
        stub.augmenters = types.ModuleType("imgaug.augmenters")
        sys.modules["imgaug"] = stub
        sys.modules["imgaug.augmenters"] = stub.augmenters
    import kitti_dataset
    import nclt_dataset
    import REIN
    for m in (kitti_dataset, nclt_dataset, REIN):
        assert Path(m.__file__).resolve().parent == OFFICIAL.resolve(), m.__file__
    spec = importlib.util.spec_from_file_location("bevplace2_main", OFFICIAL / "main.py")
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    return main, kitti_dataset, nclt_dataset, REIN


def _quiet_cv2(cv2):
    try:  # imwrite of the float64 image logs a CV_8U-fallback WARN per call
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass


def crop_paper(img: np.ndarray) -> np.ndarray:
    """201x201 code image -> 200x200 paper image (drop the never-hit row 0 /
    column 0; the per-image max normalisation is unaffected)."""
    assert img.shape == (IMG + 1, IMG + 1) and not img[0].any() and not img[:, 0].any()
    return img[1:, 1:]


def bev_image(points: np.ndarray, getBEV) -> np.ndarray:
    """Official __main__ lines 66-77 on our points (x, y, z in metres), then
    the paper's 200x200 crop."""
    pcs = np.asarray(points)[:, :3].astype(np.float32)  # .bin is float32
    pcs = pcs[np.where(np.abs(pcs[:, 0]) < 40)[0], :]
    pcs = pcs[np.where(np.abs(pcs[:, 1]) < 40)[0], :]
    pcs = pcs[np.where((np.abs(pcs[:, 2]) < 40))[0], :]
    pcs = pcs.astype(np.float32)
    img, _, _ = getBEV(pcs)
    return crop_paper(img)


def getBEV_fast(all_points):
    """Vectorised gen_bev_images.getBEV (same voxel filter, same index and
    saturation arithmetic; the per-point Python loop becomes one np.add.at).
    Used only for the on-the-fly rotation augmentation; equality with the
    official function is checked in `selftest` (including rotated clouds)."""
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(all_points)
    pc = pc.voxel_down_sample(voxel_size=0.4)
    pts = np.asarray(pc.points)
    x_max_ind = y_max_ind = np.floor(40 / 0.4).astype(int)
    x_num = y_num = x_max_ind - np.floor(-40 / 0.4).astype(int) + 1
    x_ind = x_max_ind - np.floor(pts[:, 1] / 0.4).astype(int)
    y_ind = y_max_ind - np.floor(pts[:, 0] / 0.4).astype(int)
    assert x_ind.min(initial=0) >= 0 and y_ind.min(initial=0) >= 0  # no negative wrap
    keep = (x_ind < x_num) & (y_ind < y_num)
    cnt = np.zeros((y_num, x_num), np.int64)
    np.add.at(cnt, (y_ind[keep], x_ind[keep]), 1)
    mat = np.minimum(cnt, 10).astype(np.uint8)
    mat = mat * 10
    mat[np.where(mat > 255)] = 255
    return mat / np.max(mat) * 255


def rotated_bev(points: np.ndarray, ang: float) -> np.ndarray:
    """Paper augmentation: rotate the POINT CLOUD about z (BEVPlace ICCV'23
    Sec. 5: uniform in [-pi, pi); BEVPlace++ Sec. VI-A "randomly rotating the
    point clouds"), then the official generator.  The rotation is written as
    the commented-out lines 68-70 of gen_bev_images.py (pcs.dot(rot_mat))."""
    rot_mat = np.array([[np.cos(ang), np.sin(ang), 0],
                        [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    pcs = np.asarray(points, dtype=np.float32)[:, :3].dot(rot_mat)
    pcs = pcs[np.where(np.abs(pcs[:, 0]) < 40)[0], :]
    pcs = pcs[np.where(np.abs(pcs[:, 1]) < 40)[0], :]
    pcs = pcs[np.where((np.abs(pcs[:, 2]) < 40))[0], :]
    pcs = pcs.astype(np.float32)
    return crop_paper(getBEV_fast(pcs))


def png_u8(img: np.ndarray) -> np.ndarray:
    """What cv2.imwrite stores for the float64 image (OpenCV CV_8U conversion
    = round half to even, checked in selftest), i.e. what the official loader
    reads back."""
    return np.rint(img).astype(np.uint8)


def precrop(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points)[:, :3].astype(np.float32)
    return p[(np.hypot(p[:, 0], p[:, 1]) < R_PRE) & (np.abs(p[:, 2]) < 40)]


# ------------------------------------------------------------ paper network
UPSAMPLE_MODE = "bilinear"   # paper Sec. III-B; REIN.py:132,137 use 'bicubic'
_PAPER_REM = {}


def paper_rem_class(REIN_MOD):
    """REIN.REM with its forward copied verbatim (REIN.py:78-140) except the
    interpolation of the two feature-map up-samplings (paper: "we upsample
    the feature map with bilinear interpolation").  The rotation warps
    (REIN.py:95,112) keep 'bicubic' (the paper does not specify them)."""
    if id(REIN_MOD) in _PAPER_REM:
        return _PAPER_REM[id(REIN_MOD)]
    torch, F = REIN_MOD.torch, REIN_MOD.F

    class PaperREM(REIN_MOD.REM):
        def forward(self, x):
            equ_features = []
            batch_size = x.size(0)
            for i in range(len(self.angles)):
                aff = torch.zeros(batch_size, 2, 3).cuda()
                aff[:, 0, 0] = torch.cos(-self.angles[i])
                aff[:, 0, 1] = torch.sin(-self.angles[i])
                aff[:, 1, 0] = -torch.sin(-self.angles[i])
                aff[:, 1, 1] = torch.cos(-self.angles[i])
                grid = F.affine_grid(aff, torch.Size(x.size()), align_corners=True).type(x.type())
                warped_im = F.grid_sample(x, grid, align_corners=True, mode='bicubic')
                out = self.encoder(warped_im)
                if i == 0:
                    im1_init_size = out.size()
                aff = torch.zeros(batch_size, 2, 3).cuda()
                aff[:, 0, 0] = torch.cos(self.angles[i])
                aff[:, 0, 1] = torch.sin(self.angles[i])
                aff[:, 1, 0] = -torch.sin(self.angles[i])
                aff[:, 1, 1] = torch.cos(self.angles[i])
                grid = F.affine_grid(aff, torch.Size(im1_init_size), align_corners=True).type(x.type())
                out = F.grid_sample(out, grid, align_corners=True, mode='bicubic')
                equ_features.append(out.unsqueeze(-1))
            equ_features = torch.cat(equ_features, axis=-1)  # B C H W R
            B, C, H, W, R = equ_features.shape
            equ_features = torch.max(equ_features, dim=-1, keepdim=False)[0]
            aff = torch.zeros(batch_size, 2, 3).cuda()
            aff[:, 0, 0] = 1
            aff[:, 0, 1] = 0
            aff[:, 1, 0] = 0
            aff[:, 1, 1] = 1
            B, C, H, W = x.size()
            grid = F.affine_grid(aff, torch.Size((B, C, H // 4, W // 4)), align_corners=True).type(x.type())
            out1 = F.grid_sample(equ_features, grid, align_corners=True, mode=UPSAMPLE_MODE)  # PAPER
            out1 = F.normalize(out1, dim=1)
            grid = F.affine_grid(aff, torch.Size((B, C, H, W)), align_corners=True).type(x.type())
            out2 = F.grid_sample(equ_features, grid, align_corners=True, mode=UPSAMPLE_MODE)  # PAPER
            out2 = F.normalize(out2, dim=1)
            return out1, out2

    _PAPER_REM[id(REIN_MOD)] = PaperREM
    return PaperREM


def add_netvlad_bias(model):
    """Paper soft assignment a_k(f) = softmax_k(w_k^T f + b_k) has a bias;
    REIN.py:23 builds the 1x1 conv with bias=False and init_params sets it to
    None (REIN.py:37).  A learnable bias initialised to 0 is added (with the
    code's unit-norm assignment weights the NetVLAD-paper init -alpha||c_k||^2
    is the same constant for every k, i.e. equivalent to 0)."""
    import torch
    conv = model.pooling.conv
    conv.bias = torch.nn.Parameter(torch.zeros(conv.out_channels, dtype=conv.weight.dtype,
                                               device=conv.weight.device))
    return model


def build_model(REIN_MOD):
    """REIN() (ImageNet ResNet34, as REIN.py) with the two paper overrides."""
    model = REIN_MOD.REIN()
    model.rem.__class__ = paper_rem_class(REIN_MOD)
    return add_netvlad_bias(model)


# ------------------------------------------------------------ index / gen
def cmd_index(role: str):
    seqs, _, _, pose_key = ROLES[role]
    root = BEV / role
    (root / "poses").mkdir(parents=True, exist_ok=True)
    for sensor, seq in seqs:
        name = seq_name(sensor, seq)
        cp = cache_path(role, sensor, seq)
        with np.load(cp) as z:
            sids = z["scan_ids"].astype(np.int64)
            poses = z[pose_key].astype(np.float64)
        assert len(sids) == len(poses) and np.all(np.diff(sids) > 0), name
        assert sids.max() < 10 ** 6, name   # 6-digit names keep sorted order
        d = root / name
        (d / "bev_imgs").mkdir(parents=True, exist_ok=True)
        np.save(d / "scan_ids.npy", sids)
        # KITTI pose-file format (3x4 row-major): official reads [3, 7, 11]
        np.savetxt(root / "poses" / f"{name}.txt", poses[:, :3, :].reshape(-1, 12))
        meta = {"role": role, "sensor": sensor, "sequence": seq,
                "n_frames": int(len(sids)),
                "cache": str(cp.relative_to(REPO)), "cache_sha256": sha256(cp),
                "poses_txt_from": pose_key,
                "generator": f"BEVPlace2@{OFFICIAL_COMMIT[:12]} "
                             "datasets/gen_bev_images.py getBEV (+ its __main__ "
                             "filters), cropped to the paper's 200x200, cv2.imwrite",
                "loader": "scripts/_pd_build_test_caches._make_loader"}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"OK index {role}/{name}: {len(sids)} keyframes", flush=True)


def cmd_jobs(chunk: int):
    for role, (seqs, _, _, _) in ROLES.items():
        for sensor, seq in seqs:
            sids = np.load(BEV / role / seq_name(sensor, seq) / "scan_ids.npy")
            n = max(1, -(-len(sids) // chunk))
            for k in range(n):
                print(role, f"{sensor}/{seq}", k, n)


def cmd_gen(role: str, key: str, shard: int, nshards: int):
    import cv2
    from _pd_build_test_caches import _make_loader
    _quiet_cv2(cv2)
    sensor, seq = key.split("/")
    seqs, _, _, pose_key = ROLES[role]
    assert (sensor, seq) in seqs, (role, key)
    d = BEV / role / seq_name(sensor, seq)
    sids = np.load(d / "scan_ids.npy")
    poses = np.loadtxt(BEV / role / "poses" / f"{seq_name(sensor, seq)}.txt")
    getBEV = official_getBEV()
    loader = _make_loader(sensor, seq)
    tmp = d / "_tmp"
    tmp.mkdir(exist_ok=True)
    pdir = POINTS / role / seq_name(sensor, seq)
    if role == "train":
        pdir.mkdir(parents=True, exist_ok=True)
    todo = range(shard, len(sids), nshards)
    t0, done, max_dt = time.time(), 0, 0.0
    for i in todo:
        out = d / "bev_imgs" / f"{int(sids[i]):06d}.png"
        pout = pdir / f"{int(sids[i]):06d}.npy"
        if out.exists() and (role != "train" or pout.exists()):
            continue
        item = loader[int(sids[i])]
        if item.get("pose") is not None:  # loader index space == cache index space
            dt = float(np.abs(np.asarray(item["pose"])[:3, 3] - poses[i][[3, 7, 11]]).max())
            max_dt = max(max_dt, dt)
            assert dt < 1e-2, (key, int(sids[i]), dt)
        if role == "train" and not pout.exists():
            tp = tmp / f"{int(sids[i]):06d}_{shard}.npy"
            np.save(tp, precrop(item["points"]))
            os.replace(tp, pout)
        if not out.exists():
            img = bev_image(item["points"], getBEV)
            t = tmp / f"{int(sids[i]):06d}_{shard}.png"
            assert cv2.imwrite(str(t), img), t
            os.replace(t, out)
        done += 1
        if done % 500 == 0:
            print(f"  [{role}/{key} {shard}/{nshards}] {done} "
                  f"({(time.time() - t0) / done:.3f} s/scan)", flush=True)
    print(f"OK gen {role}/{key} shard {shard}/{nshards}: {done} new in "
          f"{time.time() - t0:.0f} s, max |pose - cache| {max_dt:.2e} m", flush=True)


def cmd_status() -> bool:
    ok = True
    for role, (seqs, _, _, _) in ROLES.items():
        n_all = n_have = 0
        for sensor, seq in seqs:
            d = BEV / role / seq_name(sensor, seq)
            if not (d / "scan_ids.npy").exists():
                print(f"  {role}/{sensor}/{seq}: not indexed")
                ok = False
                continue
            sids = np.load(d / "scan_ids.npy")
            want = [f"{int(s):06d}.png" for s in sids]
            have = sorted(os.listdir(d / "bev_imgs"))
            n_all += len(want)
            n_have += len(have)
            if have != want:
                ok = False
                print(f"  {role}/{sensor}/{seq}: {len(have)}/{len(want)} PNGs")
            if have:
                import cv2
                shp = cv2.imread(str(d / "bev_imgs" / have[0]), 0).shape
                if shp != (IMG, IMG):
                    ok = False
                    print(f"  {role}/{sensor}/{seq}: image size {shp} != {(IMG, IMG)}")
            if role == "train":
                pd = POINTS / role / seq_name(sensor, seq)
                n_pts = len(os.listdir(pd)) if pd.exists() else 0
                if n_pts != len(want):
                    ok = False
                    print(f"  {role}/{sensor}/{seq}: {n_pts}/{len(want)} point files")
        print(f"{role}: {n_have}/{n_all} PNGs")
    print("BEV_COMPLETE" if ok else "BEV_INCOMPLETE")
    return ok


# ------------------------------------------------------------ CPU selftest
SELFTEST_SEQS = [("train", "kitti/01"), ("train", "helipr/Bridge01"),
                 ("val", "kitti360/0004"), ("val", "mulran/Sejong01"),
                 ("test_odom", "nclt/2012-01-08"), ("test_odom", "mcd/kth_day_06")]


def cmd_selftest(n_scans: int, out_json: Path | None):
    """(1) Adapter output vs the official script run unmodified as a
    subprocess on the same points (KITTI: the raw velodyne .bin files; other
    sensors: our loader's points stored as float32 .bin): the official 201x201
    image has an empty row 0 / column 0 and the rest equals our 200x200 image.
    (2) getBEV_fast == official getBEV on the same scans rotated by 0 and two
    random angles, with and without the 56.6 m pre-crop.  (3) the training
    augmentation path at angle 0 (pre-cropped points -> rotated_bev ->
    png_u8) reproduces the stored PNG."""
    import cv2
    from _pd_build_test_caches import _make_loader
    _quiet_cv2(cv2)
    getBEV = official_getBEV()
    rng = np.random.default_rng(0)
    report = {"official_commit": OFFICIAL_COMMIT, "cv2": cv2.__version__,
              "open3d": __import__("open3d").__version__, "cases": []}
    all_equal = True
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for role, key in SELFTEST_SEQS:
            sensor, seq = key.split("/")
            sids = np.load(cache_path(role, sensor, seq), mmap_mode="r")["scan_ids"]
            pick = [int(sids[int(i)]) for i in
                    np.linspace(0, len(sids) - 1, n_scans).round().astype(int)]
            loader = _make_loader(sensor, seq)
            vel, off, ours = (td / f"{sensor}_{seq}" / x for x in ("vel", "off", "ours"))
            for p in (vel, off, ours):
                p.mkdir(parents=True)
            times, pts_of = [], {}
            for sid in pick:
                pts = loader[sid]["points"]
                pts_of[sid] = pts
                if sensor == "kitti":
                    os.symlink(loader.frame_files[sid], vel / f"{sid:06d}.bin")
                else:
                    a = np.zeros((len(pts), 4), np.float32)
                    a[:, :3] = np.asarray(pts)[:, :3]
                    a.tofile(vel / f"{sid:06d}.bin")
                t0 = time.perf_counter()
                img = bev_image(pts, getBEV)
                times.append(time.perf_counter() - t0)
                cv2.imwrite(str(ours / f"{sid:06d}.png"), img)
            r = subprocess.run(
                [sys.executable, str(OFFICIAL / "datasets/gen_bev_images.py"),
                 "--vel_path", str(vel) + "/", "--bev_save_path", str(off) + "/"],
                capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            for sid in pick:
                a = cv2.imread(str(off / f"{sid:06d}.png"), cv2.IMREAD_UNCHANGED)
                b = cv2.imread(str(ours / f"{sid:06d}.png"), cv2.IMREAD_UNCHANGED)
                empty_edge = a.shape == (IMG + 1, IMG + 1) and not a[0].any() and not a[:, 0].any()
                same_px = b.shape == (IMG, IMG) and bool(np.array_equal(a[1:, 1:], b))
                # (2) fast == official under rotations; (3) augmentation path at 0
                pts = pts_of[sid]
                pre = precrop(pts)
                fast_eq = []
                for ang in (0.0, float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(-np.pi, np.pi))):
                    rot = np.array([[np.cos(ang), np.sin(ang), 0],
                                    [-np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
                    q = np.asarray(pts)[:, :3].astype(np.float32).dot(rot)
                    q = q[np.where(np.abs(q[:, 0]) < 40)[0], :]
                    q = q[np.where(np.abs(q[:, 1]) < 40)[0], :]
                    q = q[np.where((np.abs(q[:, 2]) < 40))[0], :]
                    ref = crop_paper(getBEV(q.astype(np.float32))[0])
                    fast_eq.append(bool(np.array_equal(ref, rotated_bev(pre, ang))))
                aug0_eq = bool(np.array_equal(png_u8(rotated_bev(pre, 0.0)), b))
                ok = empty_edge and same_px and all(fast_eq) and aug0_eq
                all_equal &= ok
                case = {"role": role, "seq": key, "scan_id": sid,
                        "official_input": "raw velodyne .bin" if sensor == "kitti"
                        else "loader points as float32 .bin",
                        "official_shape": list(a.shape), "ours_shape": list(b.shape),
                        "official_row0_col0_empty": empty_edge,
                        "pixels_equal_official_cropped": same_px,
                        "fast_getBEV_equal_official_rot": fast_eq,
                        "aug_path_angle0_equals_png": aug0_eq,
                        "nonzero_px": int((b > 0).sum()), "n_points": int(len(pts)),
                        "n_points_precrop": int(len(pre))}
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
            report.setdefault("adapter_s_per_scan", {})[key] = float(np.mean(times))
    report["all_equal"] = bool(all_equal)
    print("adapter s/scan:", json.dumps(report["adapter_s_per_scan"]))
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=1))
    print("BEV_SELFTEST_OK" if all_equal else "BEV_SELFTEST_FAILED", flush=True)
    if not all_equal:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("--role", choices=tuple(ROLES), nargs="+", default=list(ROLES))
    p = sub.add_parser("jobs")
    p.add_argument("--chunk", type=int, default=2000)
    p = sub.add_parser("gen")
    p.add_argument("--role", choices=tuple(ROLES), required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    sub.add_parser("status")
    p = sub.add_parser("selftest")
    p.add_argument("--n-scans", type=int, default=2)
    p.add_argument("--out-json", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "index":
        for r in a.role:
            cmd_index(r)
    elif a.cmd == "jobs":
        cmd_jobs(a.chunk)
    elif a.cmd == "gen":
        cmd_gen(a.role, a.seq, a.shard, a.nshards)
    elif a.cmd == "status":
        sys.exit(0 if cmd_status() else 1)
    elif a.cmd == "selftest":
        cmd_selftest(a.n_scans, a.out_json)


if __name__ == "__main__":
    main()
