"""학습 A상위10 축소와 추론 후보30 정책의 동등성을 검사한다."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _pdv2_policy_training import POLICIES, choose, restrict_graph


def test_training_and_inference_choose_identical_edges():
    """동점과 경계값에서 원래 후보 선택과 학습 그래프 축소가 같다."""
    q = np.repeat([40, 41, 42], 12)
    j = np.tile(np.arange(12), 3)
    prob = np.repeat([.8, .4, .6], 12).astype(np.float32)
    prob[0] = .9
    original = choose(q, prob, 43, "all", .6)
    ti = torch.tensor([[39, 40], [40, 39]])
    ta = torch.arange(10).reshape(2, 5).float()
    attr = torch.zeros(len(original), 5)
    attr[:, 4] = torch.from_numpy(prob[original])
    for policy in POLICIES:
        graph = Data(x=torch.zeros(43, 288), edge_index=torch.cat([ti, torch.tensor(np.stack([j[original], q[original]]))], 1),
                     edge_attr=torch.cat([ta, attr]), edge_type=torch.tensor([0, 0] + [1] * len(original)))
        restrict_graph(graph, policy, .6)
        expected = choose(q, prob, 43, policy, .6)
        np.testing.assert_array_equal(graph.edge_index[:, 2:].numpy(), np.stack([j[expected], q[expected]]))
        torch.testing.assert_close(graph.edge_index[:, :2], ti)
        torch.testing.assert_close(graph.edge_attr[:2], ta)
        assert graph.edge_attr.dtype == torch.float32 and torch.isfinite(graph.edge_attr).all()
        assert (graph.edge_type[:2] == 0).all() and (graph.edge_type[2:] == 1).all()


def test_restrict_graph_handles_no_similarity_edges():
    """유사 간선이 없어도 시간 간선은 모양과 값이 유지된다."""
    for policy in POLICIES:
        graph = Data(x=torch.zeros(3, 288), edge_index=torch.tensor([[0, 1], [1, 0]]),
                     edge_attr=torch.ones(2, 5), edge_type=torch.zeros(2, dtype=torch.long))
        before = graph.clone()
        record = restrict_graph(graph, policy, .6)
        assert record["selected"].shape == (0,)
        torch.testing.assert_close(graph.edge_index, before.edge_index)
        torch.testing.assert_close(graph.edge_attr, before.edge_attr)
