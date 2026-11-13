"""입력 정합 대조의 시퀀스 경계·자세만 교체·동일 스캔 쌍을 검사한다."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from _pdv2_input_alignment_lib import common_triplets, make_graph, pose_attributes, temporal_edges


def test_temporal_edges_match_nested_reference_without_cross_sequence():
    seq = np.r_[np.zeros(8, int), np.ones(7, int)]
    expected = [(i, j) for i in range(len(seq)) for j in range(i - 5, i + 6)
                if 0 <= j < len(seq) and j != i and seq[i] == seq[j]]
    np.testing.assert_array_equal(temporal_edges(seq), np.asarray(expected).T)


def test_pose_replacement_changes_only_attributes_and_preserves_rigid_invariance():
    seq = np.zeros(10, int)
    raw = np.arange(30, dtype=np.float32).reshape(10, 3)
    poses = np.repeat(np.eye(4)[None], 10, axis=0)
    poses[:, 0, 3] = np.arange(10)
    a = make_graph(raw, poses, seq)
    changed = poses.copy()
    changed[:, 0, 3] *= 1.1
    b = make_graph(raw, changed, seq)
    np.testing.assert_array_equal(a.x.numpy(), b.x.numpy())
    np.testing.assert_array_equal(a.edge_index.numpy(), b.edge_index.numpy())
    assert not np.array_equal(a.edge_attr.numpy(), b.edge_attr.numpy())
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [20, -30, 7]
    np.testing.assert_allclose(pose_attributes(transform @ poses, a.edge_index.numpy()),
                               a.edge_attr.numpy(), atol=1e-7)
    assert a.x.dtype.is_floating_point and np.isfinite(a.edge_attr.numpy()).all()


def test_triplets_need_all_three_scans_and_both_index_gaps():
    mapping = np.arange(150)
    mapping[140] = -1
    mapping[40] = 20
    triples = np.array([[0, 70, 100], [0, 40, 100], [0, 70, 140]])
    rows, mapped = common_triplets(triples, mapping)
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(mapped, [[0, 70, 100]])
