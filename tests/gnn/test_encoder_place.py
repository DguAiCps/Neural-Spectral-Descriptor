"""장소 제외가 특징 전달·삼중쌍에서 실제 적용되는지 검사한다."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path[:0] = [str(Path(__file__).resolve().parents[2]/'scripts'),
               str(Path(__file__).resolve().parents[2]/'src')]
from _pdv2_encoder_place_lib import restricted_triplets, temporal_subgraph


def test_all_three_triplet_nodes_must_be_in_training():
    trips = np.array([[0, 1, 2], [0, 1, 4], [4, 5, 6], [6, 4, 5]])
    np.testing.assert_array_equal(restricted_triplets(trips, np.array([4, 5, 6]), 7),
                                  [[0, 1, 2], [2, 0, 1]])


def test_heldout_nodes_and_incident_edges_are_physically_removed():
    graph = Data(x=torch.arange(15).reshape(5, 3).float(),
                 edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 3]]),
                 edge_attr=torch.arange(25).reshape(5, 5).float(),
                 edge_type=torch.tensor([0, 0, 0, 0, 1]))
    ids = np.array([2, 3, 4])
    a = temporal_subgraph(graph, ids)
    graph.x[:2] = float('nan')
    b = temporal_subgraph(graph, ids)
    assert a.num_nodes == 3 and torch.isfinite(b.x).all()
    torch.testing.assert_close(a.x, b.x)
    torch.testing.assert_close(a.edge_index, torch.tensor([[0, 1], [1, 2]]))
    torch.testing.assert_close(a.edge_attr, graph.edge_attr[2:4])
    assert a.edge_type.eq(0).all()
