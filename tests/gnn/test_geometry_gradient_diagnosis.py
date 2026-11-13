"""손실 기울기와 감쇠 진단 및 영 초기화의 전달 경로를 검사한다."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from _pdv2_geometry_gradient_lib import gradient_statistics


def test_gradient_and_decay_are_separate_and_read_only():
    p = torch.nn.Parameter(torch.tensor([3., 4.], dtype=torch.float64))
    p.grad = torch.tensor([-4., 3.], dtype=torch.float64)
    before, grad = p.detach().clone(), p.grad.clone()
    result = gradient_statistics([('gnn.edge_encoder.weight', p)], .1)['edge_encoder']
    assert result['weight_norm'] == result['loss_gradient_norm'] == 5.
    assert result['decay_gradient_norm'] == .5
    assert result['loss_to_decay_norm_ratio'] == 10.
    assert abs(result['loss_decay_cosine']) < 1e-15
    torch.testing.assert_close(p, before, atol=0, rtol=0)
    torch.testing.assert_close(p.grad, grad, atol=0, rtol=0)


def test_zero_last_weight_blocks_only_upstream_first_gradient():
    torch.manual_seed(13)
    hidden = torch.nn.Linear(2, 3, dtype=torch.float64)
    last = torch.nn.Linear(3, 1, dtype=torch.float64)
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)
    x = torch.tensor([[1., 2.], [3., 1.]], dtype=torch.float64)
    loss = (last(hidden(x).tanh()).flatten() - torch.tensor([1., -2.])).square().mean()
    loss.backward()
    assert hidden.weight.grad.abs().sum() == 0
    assert last.weight.grad.abs().sum() > 0
    assert torch.isfinite(last.weight.grad).all()


def test_adam_observation_does_not_change_step():
    torch.manual_seed(14)
    p = torch.nn.Parameter(torch.randn(7, dtype=torch.float64))
    q = torch.nn.Parameter(p.detach().clone())
    a, b = [torch.optim.Adam([v], lr=.001, weight_decay=1e-5) for v in (p, q)]
    for i in range(3):
        for v, optimizer in ((p, a), (q, b)):
            optimizer.zero_grad()
            ((v - i) ** 2).sum().backward()
        gradient_statistics([('gnn.convs.0.bias_mlp.2.weight', p)], 1e-5)
        a.step()
        b.step()
        torch.testing.assert_close(p, q, atol=0, rtol=0)
