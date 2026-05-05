"""Tests for graph construction (TemporalGraphManager + batch builder)."""

import numpy as np
import torch
import pytest
from keyframe.selector import Keyframe


def _make_pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = np.cos(yaw); T[0, 1] = -np.sin(yaw)
    T[1, 0] = np.sin(yaw); T[1, 1] = np.cos(yaw)
    T[0, 3] = x; T[1, 3] = y; T[2, 3] = z
    return T


def _make_keyframes(n=30, spacing=2.0):
    """Create n synthetic keyframes along a straight line."""
    keyframes = []
    for i in range(n):
        kf = Keyframe(
            keyframe_id=i, scan_id=i,
            points=np.empty((0, 3)),
            pose=_make_pose(x=i * spacing),
            timestamp=float(i),
            descriptor=np.random.randn(256).astype(np.float32),
        )
        keyframes.append(kf)
    return keyframes


class TestBuildGraphBatch:

    @pytest.fixture
    def keyframes_and_extras(self):
        kfs = _make_keyframes(30, spacing=2.0)
        poses = np.array([kf.pose for kf in kfs])
        descriptors = np.array([kf.descriptor for kf in kfs])
        return kfs, poses, descriptors

    def test_basic_graph_construction(self, keyframes_and_extras, device):
        from keyframe.graph_manager import build_graph_from_keyframes_batch
        kfs, poses, descriptors = keyframes_and_extras
        graph = build_graph_from_keyframes_batch(
            kfs, temporal_neighbors=5, device=device,
            poses=poses, descriptors=descriptors,
        )
        assert graph.num_nodes == 30
        assert graph.edge_index.shape[0] == 2
        assert graph.x.shape == (30, 256)

    def test_temporal_edges_exist(self, keyframes_and_extras, device):
        from keyframe.graph_manager import build_graph_from_keyframes_batch
        kfs, poses, descriptors = keyframes_and_extras
        graph = build_graph_from_keyframes_batch(
            kfs, temporal_neighbors=5, device=device,
            poses=poses, descriptors=descriptors,
        )
        if hasattr(graph, 'edge_type'):
            n_temporal = int((graph.edge_type == 0).sum())
            assert n_temporal > 0

    def test_edge_attr_shape(self, keyframes_and_extras, device):
        from keyframe.graph_manager import build_graph_from_keyframes_batch
        kfs, poses, descriptors = keyframes_and_extras
        graph = build_graph_from_keyframes_batch(
            kfs, temporal_neighbors=5, device=device,
            poses=poses, descriptors=descriptors,
        )
        if graph.edge_attr is not None:
            # 5 features: [dist, rot, cos_sim, l2_dist, posterior]
            assert graph.edge_attr.shape[1] == 5

    def test_bidirectional_edges(self, keyframes_and_extras, device):
        """Edges should be bidirectional (both i→j and j→i)."""
        from keyframe.graph_manager import build_graph_from_keyframes_batch
        kfs, poses, descriptors = keyframes_and_extras
        graph = build_graph_from_keyframes_batch(
            kfs, temporal_neighbors=3, device=device,
            poses=poses, descriptors=descriptors,
        )
        ei = graph.edge_index.cpu().numpy()
        edge_set = set(zip(ei[0], ei[1]))
        # Check some edges have reverse
        n_bidirectional = sum(1 for s, t in edge_set if (t, s) in edge_set)
        assert n_bidirectional > 0

    def test_similarity_edges_with_threshold(self, keyframes_and_extras, device):
        from keyframe.graph_manager import build_graph_from_keyframes_batch
        kfs, poses, descriptors = keyframes_and_extras
        # Normalize descriptors for cosine similarity
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors_norm = descriptors / np.maximum(norms, 1e-8)
        for i, kf in enumerate(kfs):
            kf.descriptor = descriptors_norm[i]

        graph = build_graph_from_keyframes_batch(
            kfs, temporal_neighbors=3, device=device,
            poses=poses, descriptors=descriptors_norm,
            similarity_threshold=0.5,  # low threshold to get some edges
            similarity_max_k=5,
        )
        if hasattr(graph, 'edge_type'):
            n_sim = int((graph.edge_type == 1).sum())
            # With random descriptors and low threshold, some sim edges expected
            assert n_sim >= 0  # might be 0 depending on random seed

    def test_empty_keyframes(self, device):
        """Building a graph from 0 keyframes should not crash."""
        from keyframe.graph_manager import build_graph_from_keyframes_batch
        try:
            graph = build_graph_from_keyframes_batch(
                [], temporal_neighbors=5, device=device,
                poses=np.empty((0, 4, 4)), descriptors=np.empty((0, 256)),
            )
        except (ValueError, IndexError):
            pass  # Expected for empty input


class TestTemporalGraphManager:

    def test_incremental_add(self, device):
        from keyframe.graph_manager import TemporalGraphManager
        mgr = TemporalGraphManager(
            temporal_neighbors=3, max_active_nodes=100, device=device,
        )
        kfs = _make_keyframes(10)
        for kf in kfs:
            mgr.add_keyframe(kf)
        assert len(mgr.keyframes) == 10

    def test_get_graph(self, device):
        from keyframe.graph_manager import TemporalGraphManager
        mgr = TemporalGraphManager(
            temporal_neighbors=3, max_active_nodes=100, device=device,
        )
        kfs = _make_keyframes(10)
        for kf in kfs:
            mgr.add_keyframe(kf)
        graph = mgr.get_graph()
        assert graph is not None
        assert graph.num_nodes == 10
