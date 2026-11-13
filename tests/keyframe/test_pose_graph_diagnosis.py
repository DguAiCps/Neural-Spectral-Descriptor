"""포즈 진단의 상대 변환과 현재 간선 표현의 불변성을 검산한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from _pdv2_input_alignment_lib import pose_attributes, temporal_edges
from _pdv2_pose_graph_lib import relative_geometry, ordered_windows, window_difference


def trajectory(mirror=1):
    t = np.linspace(-2, 2, 21)
    yaw = mirror * np.arctan(.4 * t)
    p = np.tile(np.eye(4), (len(t), 1, 1))
    p[:, 0, 0] = p[:, 1, 1] = np.cos(yaw)
    p[:, 1, 0], p[:, 0, 1] = np.sin(yaw), -np.sin(yaw)
    p[:, 0, 3], p[:, 1, 3] = t, mirror * .2 * t ** 2
    return p


def test_scalar_matches_independent_relative_transform():
    p = trajectory()
    e = temporal_edges(np.zeros(len(p)))
    xyz, angle = relative_geometry(p, e)
    expected = np.column_stack((np.log1p(np.linalg.norm(xyz, axis=1)) / 5, angle / np.pi))
    a = pose_attributes(p, e)
    np.testing.assert_allclose(a[:, :2], expected, atol=2e-8, rtol=0)
    assert a.dtype == np.float32 and a.shape == (180, 5) and np.isfinite(a).all()


def test_rigid_world_transform_and_reverse_edges():
    p = trajectory()
    e = temporal_edges(np.zeros(len(p)))
    world = trajectory()[3].copy()
    world[:3, 3] = [10., -20., 5.]
    np.testing.assert_allclose(pose_attributes(world @ p, e), pose_attributes(p, e), atol=2e-8)
    np.testing.assert_array_equal(pose_attributes(p, e), pose_attributes(p, e[::-1]))


def test_mirrored_turns_have_identical_scalar_attributes():
    p, mirrored = trajectory(), trajectory(-1)
    e = temporal_edges(np.zeros(len(p)))
    np.testing.assert_array_equal(pose_attributes(p, e), pose_attributes(mirrored, e))
    xyz, _ = relative_geometry(p, e)
    other, _ = relative_geometry(mirrored, e)
    assert np.max(abs(xyz[:, 1] - other[:, 1])) > .1


def test_boundaries_sequence_separation_and_window_alignment():
    seq = np.repeat([0, 1], [21, 13])
    e = temporal_edges(seq)
    assert np.all(seq[e[0]] == seq[e[1]])
    assert np.all((abs(e[0] - e[1]) >= 1) & (abs(e[0] - e[1]) <= 5))
    assert e.shape == (2, 280)
    centers, distance, angle = ordered_windows(trajectory())
    assert distance.shape == angle.shape == (11, 10)
    values = window_difference(distance, angle, centers, centers)
    assert all(np.all(v == 0) for v in values.values())
