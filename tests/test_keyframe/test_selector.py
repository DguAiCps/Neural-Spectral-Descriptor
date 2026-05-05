"""Tests for keyframe selector and criteria."""

import numpy as np
import pytest
from keyframe.selector import Keyframe, KeyframeSelector


def _make_pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = np.cos(yaw); T[0, 1] = -np.sin(yaw)
    T[1, 0] = np.sin(yaw); T[1, 1] = np.cos(yaw)
    T[0, 3] = x; T[1, 3] = y; T[2, 3] = z
    return T


class TestKeyframeDataclass:

    def test_fields_default_none(self):
        kf = Keyframe(
            keyframe_id=0, scan_id=0,
            points=np.zeros((10, 3)), pose=np.eye(4), timestamp=0.0,
        )
        assert kf.descriptor is None
        assert kf.embedding is None
        assert kf.spectral_entropy is None
        assert kf.fft_magnitudes is None

    def test_fields_set(self):
        kf = Keyframe(
            keyframe_id=1, scan_id=5,
            points=np.zeros((10, 3)), pose=np.eye(4), timestamp=1.0,
            descriptor=np.ones(1106),
            fft_magnitudes=np.ones((79, 181)),
        )
        assert kf.descriptor.shape == (1106,)
        assert kf.fft_magnitudes.shape == (79, 181)


class TestKeyframeSelector:

    @pytest.fixture
    def selector(self):
        return KeyframeSelector(
            distance_threshold=0.8,
            rotation_threshold=20.0,
            overlap_threshold=0.65,
            temporal_threshold=5.0,
        )

    def test_first_scan_always_keyframe(self, selector, random_point_cloud):
        selected, kf, details = selector.process_scan(
            scan_id=0, points=random_point_cloud,
            pose=np.eye(4), timestamp=0.0,
        )
        assert selected is True
        assert kf is not None
        assert details['reason'] == 'First keyframe'

    def test_close_scan_not_selected(self, selector, random_point_cloud):
        """A scan very close to the last keyframe should not be selected."""
        selector.process_scan(0, random_point_cloud, _make_pose(0, 0), 0.0)
        selected, kf, _ = selector.process_scan(
            1, random_point_cloud, _make_pose(0.1, 0), 0.1,
        )
        assert selected is False
        assert kf is None

    def test_distance_triggers_selection(self, selector, random_point_cloud):
        """Moving > distance_threshold should trigger keyframe."""
        selector.process_scan(0, random_point_cloud, _make_pose(0, 0), 0.0)
        selected, kf, details = selector.process_scan(
            1, random_point_cloud, _make_pose(2.0, 0), 0.5,
        )
        assert selected is True
        assert details['distance']['satisfied']

    def test_rotation_triggers_selection(self, selector, random_point_cloud):
        """Rotating > rotation_threshold should trigger keyframe."""
        selector.process_scan(0, random_point_cloud, _make_pose(0, 0, yaw=0), 0.0)
        selected, _, details = selector.process_scan(
            1, random_point_cloud,
            _make_pose(0.1, 0, yaw=np.radians(25)), 0.5,
        )
        assert selected is True
        assert details['rotation']['satisfied']

    def test_temporal_triggers_selection(self, selector, random_point_cloud):
        """Time gap > temporal_threshold should trigger keyframe."""
        selector.process_scan(0, random_point_cloud, _make_pose(0, 0), 0.0)
        selected, _, details = selector.process_scan(
            1, random_point_cloud, _make_pose(0.1, 0), 10.0,
        )
        assert selected is True
        assert details['temporal']['satisfied']

    def test_or_logic(self, selector, random_point_cloud):
        """Any single criterion triggers selection (OR logic)."""
        selector.process_scan(0, random_point_cloud, _make_pose(0, 0), 0.0)
        # Only temporal is satisfied (close position, no rotation, big time gap)
        selected, _, _ = selector.process_scan(
            1, random_point_cloud, _make_pose(0.1, 0), 100.0,
        )
        assert selected is True

    def test_reset(self, selector, random_point_cloud):
        selector.process_scan(0, random_point_cloud, np.eye(4), 0.0)
        assert len(selector.keyframes) == 1
        selector.reset()
        assert len(selector.keyframes) == 0
        assert selector.last_keyframe is None

    def test_max_keyframes_limit(self, random_point_cloud):
        selector = KeyframeSelector(
            distance_threshold=0.1, max_keyframes=5
        )
        for i in range(20):
            selector.process_scan(
                i, random_point_cloud,
                _make_pose(x=i * 5.0), float(i),
            )
        assert len(selector.keyframes) <= 5

    def test_keyframe_id_increments(self, selector, random_point_cloud):
        for i in range(5):
            selector.process_scan(
                i, random_point_cloud,
                _make_pose(x=i * 5.0), float(i) * 10,
            )
        ids = [kf.keyframe_id for kf in selector.keyframes]
        assert ids == list(range(len(ids)))

    def test_get_keyframe_by_id(self, selector, random_point_cloud):
        selector.process_scan(0, random_point_cloud, np.eye(4), 0.0)
        kf = selector.get_keyframe_by_id(0)
        assert kf is not None
        assert kf.keyframe_id == 0

    def test_get_keyframe_by_scan_id(self, selector, random_point_cloud):
        selector.process_scan(5, random_point_cloud, np.eye(4), 0.0)
        kf = selector.get_keyframe_by_scan_id(5)
        assert kf is not None
        assert kf.scan_id == 5

    def test_statistics(self, selector, random_point_cloud):
        for i in range(10):
            selector.process_scan(
                i, random_point_cloud,
                _make_pose(x=i * 5.0), float(i) * 10,
            )
        stats = selector.get_statistics()
        assert stats['num_scans'] == 10
        assert stats['num_keyframes'] > 0
        assert stats['compression_ratio'] > 0

    def test_attach_descriptors(self, selector, random_point_cloud):
        for i in range(3):
            selector.process_scan(
                i, random_point_cloud,
                _make_pose(x=i * 5.0), float(i) * 10,
            )
        n_kf = len(selector.keyframes)
        descs = np.random.randn(n_kf, 1106).astype(np.float32)
        selector.attach_descriptors(descs)
        for kf in selector.keyframes:
            assert kf.descriptor is not None

    def test_attach_embeddings(self, selector, random_point_cloud):
        selector.process_scan(0, random_point_cloud, np.eye(4), 0.0)
        emb = np.random.randn(1, 512).astype(np.float32)
        selector.attach_embeddings(emb)
        assert selector.keyframes[0].embedding is not None

    def test_export_poses(self, selector, random_point_cloud):
        for i in range(3):
            selector.process_scan(
                i, random_point_cloud,
                _make_pose(x=i * 5.0), float(i) * 10,
            )
        poses = selector.export_keyframe_poses()
        assert poses.shape[0] == len(selector.keyframes)
        assert poses.shape[1:] == (4, 4)

    def test_process_sequence(self, random_point_cloud):
        selector = KeyframeSelector(distance_threshold=0.8, temporal_threshold=5.0)
        points_list = [random_point_cloud for _ in range(10)]
        poses = np.array([_make_pose(x=i * 5.0) for i in range(10)])
        timestamps = np.arange(10) * 10.0
        keyframes = selector.process_sequence(points_list, poses, timestamps)
        assert len(keyframes) > 0
        assert len(keyframes) == len(selector.keyframes)
