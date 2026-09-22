"""Equivalence tests for the vectorized ImLPR RIV preprocessing.

Verifies scripts/_imlpr_riv_lib.compute_normals_fast against the OFFICIAL
per-point compute_normals from _external/ImLPR/preprocess/generate_riv_images.py
on real frames from two sensor families (dense 64-beam KITTI HDL-64E and
sparse HeLiPR VLP-16), plus v-index and intensity-scaling sanity checks.

The real-frame tests need the raw dataset mount and are meant to run inside
docker ring_baseline:

    docker exec ring_baseline bash -c \
      "cd /workspace/Neural-Spectral-Codec && python3 -m pytest -q \
       tests/test_imlpr_riv_equiv.py"

Outside the container (no /workspace/data) the data-dependent tests skip.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

lib = pytest.importorskip(
    "_imlpr_riv_lib",
    reason="ImLPR RIV deps (cv2/sklearn/open3d) unavailable on this host")
riv_official = lib.riv_official

N_SUB = 2000  # official loop is O(N) Python SVDs; compare on a 2000-pt subcloud

# (sensor, sequence): one dense 64-beam + the VLP-16 stream
REAL_FRAMES = [("kitti", "05"), ("helipr", "Town01")]


def _real_subcloud(sensor: str, seq: str) -> np.ndarray:
    """Load a real roster keyframe and subsample N_SUB xyz points (fixed seed)."""
    entry = next(e for e in lib.load_roster()
                 if e["sensor"] == sensor and e["sequence"] == seq)
    scan_ids = lib.load_cache_scan_ids(entry)
    loader = lib.make_loader(sensor, seq)
    pts = loader[int(scan_ids[len(scan_ids) // 2])]["points"][:, :3]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=min(N_SUB, len(pts)), replace=False)
    return np.ascontiguousarray(pts[idx], dtype=np.float32)


needs_data = pytest.mark.skipif(
    not lib.DATA_ROOT.is_dir(),
    reason=f"raw dataset root {lib.DATA_ROOT} unavailable "
           "(run inside docker ring_baseline)")


@needs_data
@pytest.mark.parametrize("sensor,seq", REAL_FRAMES)
def test_normals_equivalence_real_frame(sensor, seq):
    """Official per-point SVD loop == vectorized eigvalsh, on real points.

    Both run on the SAME subsampled cloud (identical sklearn kNN structure),
    so only the SVD-vs-scatter-eigvalsh math differs.
    """
    sub = _real_subcloud(sensor, seq)

    off = riv_official.compute_normals(
        sub, k=lib.K_NEIGHBORS, max_neighbor_m=lib.MAX_NEIGHBOR_M)
    fast = lib.compute_normals_fast(
        sub, k=lib.K_NEIGHBORS, max_neighbor_m=lib.MAX_NEIGHBOR_M)

    assert off.shape == fast.shape == (len(sub),)
    assert np.allclose(fast, off, rtol=1e-3, atol=1e-3), (
        f"max abs diff {np.abs(fast - off).max()}")


@needs_data
@pytest.mark.parametrize("sensor,seq", REAL_FRAMES)
def test_normal_channel_equivalence_real_frame(sensor, seq):
    """The rendered (uint8-scaled) normal channel matches through the official
    renderer as well."""
    sub = _real_subcloud(sensor, seq)
    cfg = lib.SENSORS[sensor]
    ring = np.zeros(len(sub), dtype=np.int32)

    off = riv_official.compute_normals(
        sub, k=lib.K_NEIGHBORS, max_neighbor_m=lib.MAX_NEIGHBOR_M)
    fast = lib.compute_normals_fast(
        sub, k=lib.K_NEIGHBORS, max_neighbor_m=lib.MAX_NEIGHBOR_M)

    kw = dict(image_width=lib.IMAGE_WIDTH, image_height_initial=cfg["H0"],
              image_height_final=None, v_mode=cfg["v_mode"],
              fov_up_deg=cfg["fov_up_deg"], fov_down_deg=cfg["fov_down_deg"],
              invert_vertical=False)
    img_off = riv_official.pointCloudToImage(sub, off, ring, **kw)
    img_fast = riv_official.pointCloudToImage(sub, fast, ring, **kw)

    assert np.allclose(img_fast, img_off, rtol=1e-3, atol=1e-3)
    assert np.abs(img_fast.astype(np.uint8).astype(np.int16)
                  - img_off.astype(np.uint8).astype(np.int16)).max() <= 1


def test_normals_far_neighbor_rule():
    """Isolated points (neighbor > max_neighbor_m) get the official 0.7 default
    in both implementations."""
    rng = np.random.default_rng(1)
    cluster = rng.normal(size=(200, 3)).astype(np.float32)
    lonely = np.array([[500.0, 500.0, 500.0]], dtype=np.float32)
    pts = np.concatenate([cluster, lonely])

    off = riv_official.compute_normals(pts, k=4, max_neighbor_m=50.0)
    fast = lib.compute_normals_fast(pts, k=4, max_neighbor_m=50.0)
    expected = riv_official.scale_log_to_255([0.7])[0]
    assert off[-1] == pytest.approx(expected)
    assert fast[-1] == pytest.approx(expected)
    assert np.allclose(fast, off, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("sensor", ["kitti", "nclt", "helipr", "mcd"])
def test_v_indices_angle_mode_maps_fov_up_to_row0(sensor):
    """Angle mode: elevation +fov_up -> row 0 (top), -fov_down -> row H0-1."""
    cfg = lib.SENSORS[sensor]
    up, down, H0 = cfg["fov_up_deg"], cfg["fov_down_deg"], cfg["H0"]

    def pt(elev_deg):
        e = np.radians(elev_deg)
        return [np.cos(e), 0.0, np.sin(e)]

    pts = np.array([pt(up), pt(-down), pt((up - down) / 2.0)], dtype=np.float32)
    v = riv_official.compute_v_indices(
        pts, None, H0, mode="angle", fov_up_deg=up, fov_down_deg=down)
    assert v[0] == 0
    # official formula divides by total_fov + 1e-6 and truncates to int32, so
    # the exact bottom-of-FOV elevation lands on the last or second-to-last row
    assert H0 - 2 <= v[1] <= H0 - 1
    assert 0 < v[2] < H0 - 1  # mid-FOV elevation lands strictly inside


def test_intensity_scaling_per_sensor_family():
    # kitti/kitti360: loader floats 0..1 -> x255
    assert np.allclose(lib.intensity_to_255(np.array([0.0, 0.5, 1.0]), "kitti"),
                       [0.0, 127.5, 255.0])
    assert np.allclose(lib.intensity_to_255(np.array([1.0]), "kitti360"), [255.0])
    # nclt: NCLTLoader returns raw_uint8/255 -> x255 restores raw 0..255
    assert np.allclose(lib.intensity_to_255(np.array([37 / 255.0]), "nclt"), [37.0])
    # helipr VLP-16: already 0..255, clip only
    assert np.allclose(lib.intensity_to_255(np.array([-3.0, 100.0, 300.0]),
                                            "helipr"), [0.0, 100.0, 255.0])
    # mulran/mcd: official read_bin_file mulran branch (clamp 4096, /4096*255)
    assert np.allclose(lib.intensity_to_255(np.array([0.0, 2048.0, 4096.0, 8000.0]),
                                            "mulran"), [0.0, 127.5, 255.0, 255.0])
    assert np.allclose(lib.intensity_to_255(np.array([4096.0, 9999.0]), "mcd"),
                       [255.0, 255.0])
