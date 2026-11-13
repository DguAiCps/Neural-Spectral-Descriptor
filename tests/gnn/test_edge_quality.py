"""간선 제거 대조의 개수·정보 차단·부분 입력 일치를 검사한다."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from _pdv2_edge_quality import masks, filter_graph, edge_counts


def test_masks_counts_prefix_and_gt_isolation():
    q = np.repeat(np.arange(40, 44), 10)
    j = np.tile(np.arange(10), 4)
    p = np.linspace(0, 1, 40, dtype=np.float32)
    truth = np.arange(40) % 3 == 0
    m = masks(q, j, p, truth)
    other = masks(q, j, p, ~truth)
    cut = masks(q[:20], j[:20], p[:20], truth[:20])
    for name in m:
        assert m[name].dtype == np.bool_ and m[name].shape == q.shape
        np.testing.assert_array_equal(m[name][:20], cut[name])
        if name != "oracle_true":
            np.testing.assert_array_equal(m[name], other[name])
    np.testing.assert_array_equal(np.bincount(q[m["p050"]], minlength=44),
                                  np.bincount(q[m["random050"]], minlength=44))
    assert m["p000"].all() and not m["temporal"].any()
    assert np.all(m["p090"] <= m["p075"]) and np.all(m["p075"] <= m["p050"])
    assert all(len(x) == 0 for x in masks(q[:0], j[:0], p[:0], truth[:0]).values())


def test_filter_preserves_time_attributes_and_original_graph():
    g = Data(x=torch.ones(44, 288), edge_index=torch.tensor([[38, 0, 39, 1], [39, 40, 40, 41]]),
             edge_type=torch.tensor([0, 1, 0, 1]), edge_attr=torch.arange(20).float().reshape(4, 5))
    before = g.clone()
    out = filter_graph(g, np.array([False, True]))
    assert out.edge_index.shape == (2, 3) and torch.isfinite(out.edge_attr).all()
    assert torch.equal(out.edge_attr, g.edge_attr[[0, 2, 3]])
    assert torch.equal(g.edge_index, before.edge_index) and torch.equal(g.edge_attr, before.edge_attr)
    assert torch.equal(filter_graph(g, np.array([False, False])).edge_index, g.edge_index[:, [0, 2]])


def test_quality_distances_and_denominators():
    d = edge_counts(np.array([0, 1, 1, 2]), np.array([1., 5., 25., 30.]),
                    np.array([1., 4., 20., 30.]), np.array([.1, .5, .6, .9]), np.array([False, True, True]))
    assert d["edges"] == 3 and d["nodes"] == 2
    assert d["positive_3d"] == d["ambiguous_3d"] == d["far_3d"] == 1
    assert d["nodes_with_edge"] == 2 and d["nodes_with_true_edge"] == 1
