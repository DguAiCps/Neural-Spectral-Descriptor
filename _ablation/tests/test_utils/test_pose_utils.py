"""Tests for SE(3) pose utilities."""

import numpy as np
import pytest
from data.pose_utils import (
    euclidean_distance,
    rotation_angle,
    rotation_angle_degrees,
    inverse_pose,
    compose_poses,
    relative_pose,
    transform_points,
    is_valid_transformation,
    cartesian_to_spherical,
    spherical_to_cartesian,
    pose_difference,
    interpolate_poses,
)


class TestEuclideanDistance:

    def test_zero_distance(self):
        T = np.eye(4)
        assert euclidean_distance(T, T) == pytest.approx(0.0)

    def test_known_distance(self):
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[0, 3] = 3.0
        T2[1, 3] = 4.0
        assert euclidean_distance(T1, T2) == pytest.approx(5.0)

    def test_symmetric(self):
        T1 = np.eye(4)
        T1[0, 3] = 1.0
        T2 = np.eye(4)
        T2[1, 3] = 2.0
        assert euclidean_distance(T1, T2) == pytest.approx(euclidean_distance(T2, T1))


class TestRotationAngle:

    def test_zero_rotation(self):
        T = np.eye(4)
        assert rotation_angle(T, T) == pytest.approx(0.0, abs=1e-6)

    def test_90_degree_rotation(self):
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        angle = rotation_angle_degrees(T1, T2)
        assert angle == pytest.approx(90.0, abs=0.5)

    def test_180_degree_rotation(self):
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        angle = rotation_angle_degrees(T1, T2)
        assert angle == pytest.approx(180.0, abs=0.5)


class TestInversePose:

    def test_inverse_identity(self):
        T_inv = inverse_pose(np.eye(4))
        np.testing.assert_allclose(T_inv, np.eye(4), atol=1e-10)

    def test_inverse_compose_identity(self):
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        result = compose_poses(T, inverse_pose(T))
        np.testing.assert_allclose(result, np.eye(4), atol=1e-10)


class TestRelativePose:

    def test_relative_pose_identity(self):
        T = np.eye(4)
        T[:3, 3] = [5, 0, 0]
        T_rel = relative_pose(T, T)
        np.testing.assert_allclose(T_rel, np.eye(4), atol=1e-10)

    def test_relative_pose_translation(self):
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[0, 3] = 10.0
        T_rel = relative_pose(T1, T2)
        assert T_rel[0, 3] == pytest.approx(10.0)


class TestTransformPoints:

    def test_identity_transform(self):
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = transform_points(pts, np.eye(4))
        np.testing.assert_allclose(result, pts, atol=1e-10)

    def test_translation(self):
        pts = np.array([[0, 0, 0]], dtype=np.float64)
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]
        result = transform_points(pts, T)
        np.testing.assert_allclose(result, [[1, 2, 3]], atol=1e-10)

    def test_4d_keeps_intensity(self):
        pts = np.array([[1, 0, 0, 0.5]], dtype=np.float64)
        T = np.eye(4)
        T[0, 3] = 10.0
        result = transform_points(pts, T)
        assert result[0, 3] == pytest.approx(0.5)
        assert result[0, 0] == pytest.approx(11.0)


class TestValidation:

    def test_identity_is_valid(self):
        assert is_valid_transformation(np.eye(4)) is True

    def test_wrong_shape(self):
        assert is_valid_transformation(np.eye(3)) is False

    def test_bad_bottom_row(self):
        T = np.eye(4)
        T[3, 0] = 1.0
        assert is_valid_transformation(T) is False

    def test_reflection_invalid(self):
        T = np.eye(4)
        T[0, 0] = -1  # reflection
        assert is_valid_transformation(T) is False


class TestSphericalConversion:

    def test_roundtrip(self):
        pts = np.array([[10, 5, 3], [-2, 7, -1]], dtype=np.float64)
        spherical = cartesian_to_spherical(pts)
        reconstructed = spherical_to_cartesian(spherical)
        np.testing.assert_allclose(reconstructed, pts, atol=1e-10)


class TestPoseDifference:

    def test_identity_difference(self):
        trans, rot = pose_difference(np.eye(4), np.eye(4))
        assert trans == pytest.approx(0.0)
        assert rot == pytest.approx(0.0, abs=1e-6)


class TestInterpolatePoses:

    def test_endpoints(self):
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[0, 3] = 10.0
        T0 = interpolate_poses(T1, T2, 0.0)
        T1_end = interpolate_poses(T1, T2, 1.0)
        np.testing.assert_allclose(T0[:3, 3], T1[:3, 3], atol=1e-6)
        np.testing.assert_allclose(T1_end[:3, 3], T2[:3, 3], atol=1e-6)

    def test_midpoint(self):
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[0, 3] = 10.0
        T_mid = interpolate_poses(T1, T2, 0.5)
        assert T_mid[0, 3] == pytest.approx(5.0, abs=0.1)
