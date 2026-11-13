"""상쇄량의 정의와 시간 방향 분해를 독립 합산으로 검산한다."""
import torch

from gnn.message_cancellation import cancellation_statistics


def test_opposite_messages_and_one_sided_messages():
    edge = torch.tensor([[0, 2, 3], [1, 1, 4]])
    messages = torch.tensor([[2., -1.], [-2., 1.], [3., -2.]], dtype=torch.float64)
    r = cancellation_statistics(messages, edge, 5)
    assert r["cancellation"][1] == r["between"][1] == 1
    assert r["within"][1] == 0
    assert abs(float(r["side_cosine"][1]) + 1) < 1e-14
    assert r["cancellation"][4] == r["between"][4] == 0
    assert not r["valid"][0] and not r["both_sides"][4]


def test_independent_loop_and_mass_decomposition():
    torch.manual_seed(831)
    edge = torch.tensor([[0, 1, 3, 4, 0, 2, 4, 5], [2, 2, 2, 2, 3, 3, 3, 3]])
    message = torch.randn(8, 16, dtype=torch.float64)
    r = cancellation_statistics(message, edge, 6)
    for node in (2, 3):
        vectors = [v for v, target in zip(message, edge[1]) if target == node]
        expected = sum(vectors)
        mass = sum(float(v.abs().sum()) for v in vectors)
        torch.testing.assert_close(r["total"][node], expected, atol=1e-14, rtol=0)
        assert abs(float(r["cancellation"][node]) - (1 - float(expected.abs().sum()) / mass)) < 1e-14
    torch.testing.assert_close(r["cancellation"], r["within"] + r["between"], atol=1e-14, rtol=0)
    for value in r.values():
        assert torch.isfinite(value).all()


def test_scaling_and_channel_sign_invariance():
    torch.manual_seed(83)
    edge = torch.tensor([[0, 2, 3, 1, 3], [1, 1, 1, 2, 2]])
    message = torch.randn(5, 4, dtype=torch.float64)
    a = cancellation_statistics(message, edge, 4)
    b = cancellation_statistics(7 * message * torch.tensor([1., -1., 1., -1.]), edge, 4)
    for key in ("cancellation", "within", "between", "side_cosine"):
        torch.testing.assert_close(a[key], b[key], atol=1e-14, rtol=0)
