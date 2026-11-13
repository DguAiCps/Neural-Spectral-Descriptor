"""거리 라벨이 빠진 축·좌표계 변환·경계에 올바르게 반응하는지 검사한다."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from retrieval.edge_labels import edge_labels_3d
from _pdv2_edge_labels3d import replace_edges


def test_third_axis_and_distance_boundaries():
    p = np.array([[0., 0., 0.], [0., 0., 4.], [0., 0., 5.],
                  [0., 0., 24.], [0., 0., 25.], [.9, 0., 43.]])
    labels = edge_labels_3d(p, np.zeros(5, np.int64), np.arange(1, 6))
    np.testing.assert_array_equal(labels, [1, -1, -1, 0, 0])
    assert labels.dtype == np.int8 and labels.shape == (5,)
    assert edge_labels_3d(p, np.array([], np.int64), np.array([], np.int64)).shape == (0,)
    with pytest.raises(ValueError):
        edge_labels_3d(p[:, :2], np.array([0]), np.array([1]))


def test_labels_do_not_depend_on_coordinate_axes_or_origin():
    rng = np.random.default_rng(81)
    p = rng.normal(size=(40, 3)) * 20
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q, j = np.arange(20), np.arange(20, 40)
    before = edge_labels_3d(p, q, j)
    np.testing.assert_array_equal(before, edge_labels_3d(p @ rotation + [200., -80., 7.], q, j))


def test_edge_replacement_keeps_time_edges_and_original_input():
    graph = Data(x=torch.ones(45, 288), edge_index=torch.tensor([[38, 0, 39], [39, 40, 40]]),
                 edge_type=torch.tensor([0, 1, 0]), edge_attr=torch.arange(15).float().reshape(3, 5))
    original = graph.clone()
    c = {"query": np.array([40, 40, 41]), "neighbor": np.array([0, 1, 2]),
         "features": np.ones((3, 14), np.float32) * .8, "l2": np.array([1., 2., 3.], np.float32)}
    result = replace_edges(graph, c, np.array([.1, .9, .3], np.float32), np.array([1, 2]))
    np.testing.assert_array_equal(result.edge_index.numpy(), [[38, 39, 1, 2], [39, 40, 40, 41]])
    assert result.edge_attr.dtype == torch.float32 and torch.isfinite(result.edge_attr).all()
    assert torch.equal(result.edge_attr[:2], original.edge_attr[[0, 2]])
    assert torch.equal(result.x, original.x) and torch.equal(graph.edge_attr, original.edge_attr)
    assert torch.equal(graph.edge_index, original.edge_index)
