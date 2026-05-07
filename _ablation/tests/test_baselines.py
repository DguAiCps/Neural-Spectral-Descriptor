"""
Invariance unit tests for the faithful baseline reimplementations.

Each test rotates a synthetic structured point cloud and verifies that the
baseline's intended invariance mechanism actually delivers near-perfect
similarity post-rotation.
"""

import numpy as np
import pytest


@pytest.fixture
def structured_cloud():
    """Asymmetric building-corner cloud: not isotropic, so PCA/col-shift is testable."""
    rng = np.random.default_rng(0)
    n = 8000
    pts = np.empty((n, 3), dtype=np.float32)
    pts[:, 0] = rng.uniform(-30, 30, n)
    pts[:, 1] = rng.uniform(-15, 15, n)
    pts[:, 2] = rng.uniform(-1, 5, n)
    return pts


def _yaw_rotate(pts, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=pts.dtype)
    out = pts.copy()
    out[:, :3] = pts[:, :3] @ R.T
    return out


def test_m2dp_pca_yaw_invariance(structured_cloud):
    """PCA pose normalization gives near-perfect rotation invariance."""
    from baselines.m2dp import M2DP
    m = M2DP()
    d1 = m.encode(structured_cloud)
    d2 = m.encode(_yaw_rotate(structured_cloud, np.pi / 4))
    assert float(np.dot(d1, d2)) > 0.99, "M2DP+PCA should be ~exactly yaw-invariant"


def test_m2dp_no_pca_is_not_invariant(structured_cloud):
    """Sanity: without PCA, M2DP is not exactly yaw-invariant."""
    from baselines.m2dp import M2DP
    m = M2DP(pca_pose_normalization=False)
    d1 = m.encode(structured_cloud)
    d2 = m.encode(_yaw_rotate(structured_cloud, np.pi / 4))
    cos = float(np.dot(d1, d2))
    assert cos < 0.99, f"Without PCA, M2DP should not be perfectly invariant (got {cos:.4f})"


def test_sc_pp_columnshift_invariance(structured_cloud):
    """SC matrix col-shift cosine distance is ~0 for any yaw rotation."""
    from baselines.scan_context import build_scan_context, _distance_sc_columnwise
    sc1 = build_scan_context(structured_cloud)
    sc2 = build_scan_context(_yaw_rotate(structured_cloud, np.pi / 2))
    d = _distance_sc_columnwise(sc1, sc2)
    assert d < 0.05, f"SC col-shift distance should be ~0 (got {d:.4f})"


def test_fresco_yaw_translation_invariance(structured_cloud):
    """Fourier-Mellin → joint yaw + translation invariance."""
    from baselines.fresco import FreSCo
    f = FreSCo()
    d0 = f.encode(structured_cloud)
    d_rot = f.encode(_yaw_rotate(structured_cloud, np.pi / 2))
    d_trans = f.encode(structured_cloud + np.array([5.0, 0.0, 0.0], dtype=np.float32))
    assert float(np.dot(d0, d_rot)) > 0.99, "FreSCo yaw invariance"
    assert float(np.dot(d0, d_trans)) > 0.95, "FreSCo translation invariance"


def test_lidar_iris_hamming_yaw_invariance(structured_cloud):
    """Min-shift Hamming distance over column shifts is ~0 for yaw rotation."""
    from baselines.lidar_iris import (
        _build_iris_8bit, _make_templates, _hamming_min_shift
    )
    t1 = _make_templates(_build_iris_8bit(structured_cloud))
    t2 = _make_templates(_build_iris_8bit(_yaw_rotate(structured_cloud, np.pi / 2)))
    d = _hamming_min_shift(t1, t2, max_shift=180)
    assert d < 0.05, f"LiDAR-Iris min-shift Hamming should be ~0 (got {d:.4f})"


def test_lidar_iris_descriptor_dim():
    """Iris coarse signature is 640D as documented."""
    from baselines.lidar_iris import LiDARIris
    iris = LiDARIris()
    assert iris.descriptor_dim == 640, f"Expected 640D, got {iris.descriptor_dim}"


def test_fresco_descriptor_dim():
    """FreSCo descriptor stays 620D after Fourier-Mellin."""
    from baselines.fresco import FreSCo
    f = FreSCo()
    assert f.descriptor_dim == 620, f"Expected 620D, got {f.descriptor_dim}"
