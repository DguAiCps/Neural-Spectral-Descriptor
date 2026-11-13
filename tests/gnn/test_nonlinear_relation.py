"""선형 상쇄·비선형 관계·가중치 보존·역전파의 결정적 검산."""
import torch

from gnn.model import DiffAttnConv, create_spectral_gnn
from gnn.nonlinear_relation import NonlinearRelationConv, attach_nonlinear_relation


def test_linear_cancellation_and_nonlinear_counterexample():
    x = torch.tensor([[1.], [2.], [4.]], dtype=torch.float64)
    edges = torch.tensor([[1, 2], [0, 0]])
    weights = {}
    for mode in ("diff", "standard"):
        conv = DiffAttnConv(1, heads=1, dropout=0., value_source=mode).double().eval()
        with torch.no_grad():
            for p in conv.parameters():
                p.fill_(1.)
        _, (_, weights[mode]) = conv(x, edges, return_attention_weights=True)
        nonlinear = NonlinearRelationConv(conv).eval()
        out, (_, weights[mode+"_tanh"]) = nonlinear(x, edges, return_attention_weights=True)
        assert out.shape == x.shape and out.dtype == x.dtype and torch.isfinite(out).all()
    torch.testing.assert_close(weights["diff"], weights["standard"], atol=1e-14, rtol=0)
    assert (weights["diff_tanh"]-weights["standard_tanh"]).abs().max() > .04


def test_independent_dense_attention_and_gradient():
    torch.manual_seed(472)
    x = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor([[0, 2, 3, 4, 6], [1, 1, 1, 5, 5]])
    attr = torch.randn(5, 3, dtype=torch.float64)
    logit = torch.randn(5, 1, dtype=torch.float64)
    gate = torch.rand(5, 2, dtype=torch.float64)
    conv = NonlinearRelationConv(DiffAttnConv(8, heads=2, edge_dim=3, dropout=0.)).double()
    actual = conv(x, edges, attr, logit, gate)
    expected = torch.zeros_like(x)
    for i in (1, 5):
        mask = edges[1] == i
        delta = x[edges[0, mask]]-x[i]
        query = conv.W_q(x[i]).reshape(2, 4)
        key = conv.W_k(delta).tanh().reshape(-1, 2, 4)
        score = (key*query).sum(-1)/2+conv.bias_mlp(attr[mask])+logit[mask]
        value = conv.W_v(delta).reshape(-1, 2, 4)
        expected[i] = (torch.softmax(score, dim=0)[..., None]*value*gate[mask, :, None]).sum(0).flatten()
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    actual.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in conv.parameters())


def test_parameter_and_state_identity():
    torch.manual_seed(4)
    net = create_spectral_gnn(input_dim=288, hidden_dim=128, context_dim=128,
                              n_layers=2, n_heads=4, dropout=.1, norm_type="batch_norm",
                              use_residual_gate=True, gate_hidden_dim=64,
                              gate_initial_alpha=.0625, diffattn_value_source="diff")
    before = {k: v.clone() for k, v in net.state_dict().items()}
    count = sum(p.numel() for p in net.parameters())
    attach_nonlinear_relation(net)
    assert count == sum(p.numel() for p in net.parameters())
    assert before.keys() == net.state_dict().keys()
    for k, v in before.items():
        torch.testing.assert_close(v, net.state_dict()[k], atol=0, rtol=0)


def test_empty_edges_and_permutation_equivariance():
    torch.manual_seed(8)
    x = torch.randn(5, 8, dtype=torch.float64)
    conv = NonlinearRelationConv(DiffAttnConv(8, heads=2, dropout=0.)).double().eval()
    empty = conv(x, torch.empty((2, 0), dtype=torch.long))
    torch.testing.assert_close(empty, torch.zeros_like(x))
    edge = torch.tensor([[0, 1, 2, 3], [2, 2, 4, 4]])
    perm = torch.tensor([4, 2, 0, 3, 1])
    inv = torch.argsort(perm)
    a, (_, weight) = conv(x, edge, return_attention_weights=True)
    b = conv(x[perm], inv[edge])
    torch.testing.assert_close(a[perm], b, atol=1e-13, rtol=0)
    assert torch.isfinite(weight).all()


def test_autocast_message_accumulation_and_backward():
    torch.manual_seed(9)
    x = torch.randn(5, 8, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 3], [2, 2, 4, 4]])
    conv = NonlinearRelationConv(DiffAttnConv(8, heads=2, dropout=0.))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = conv(x, edges)
        loss = output.float().square().sum()
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    loss.backward()
    assert torch.isfinite(x.grad).all()
