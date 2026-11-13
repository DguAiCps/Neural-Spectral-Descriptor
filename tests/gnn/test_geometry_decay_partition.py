"""기하 분기 감쇠 분리의 적용 범위와 동등 감쇠 시 Adam 일치를 검사한다."""
import copy
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from _pdv2_geometry_gradient_lib import optimizer_partition, parameter_group


def network():
    net = nn.Module()
    net.gnn = nn.Module()
    net.gnn.edge_encoder = nn.Linear(2, 3)
    net.gnn.convs = nn.ModuleList([nn.Module(), nn.Module()])
    for conv in net.gnn.convs:
        conv.bias_mlp = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 2))
        conv.W_q = nn.Linear(2, 2)
    net.gnn.output_proj = nn.Linear(2, 2)
    return net


def test_decay_is_removed_only_from_geometry():
    net = network()
    groups = optimizer_partition(net, 0.)
    membership = {id(p): g['weight_decay'] for g in groups for p in g['params']}
    assert len(membership) == len(list(net.parameters()))
    for name, parameter in net.named_parameters():
        geometry = parameter_group(name).startswith(('edge', 'bias'))
        assert membership[id(parameter)] == (0. if geometry else 1e-5)


def test_partition_alone_preserves_adam_updates():
    torch.manual_seed(11)
    a = network()
    b = copy.deepcopy(a)
    oa = torch.optim.Adam(a.parameters(), lr=.001, weight_decay=1e-5)
    ob = torch.optim.Adam(optimizer_partition(b, 1e-5), lr=.001)
    for _ in range(3):
        for pa, pb in zip(a.parameters(), b.parameters()):
            gradient = torch.randn_like(pa)
            pa.grad, pb.grad = gradient.clone(), gradient.clone()
        oa.step()
        ob.step()
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, atol=0, rtol=0)
