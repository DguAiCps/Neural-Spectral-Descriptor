"""최종 판정의 관측 범위와 음성 질의 기울기를 검사한다."""
import numpy as np
import torch

from retrieval.nonrevisit_learning import acceptance_loss, context_plan, context_score
from retrieval.sequence_evidence import neighbor_plans, path_means, repeated_candidate_weights


def test_context_score_matches_numpy_and_prefix():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 416)).astype(np.float32)
    path = np.r_[0., np.cumsum(rng.uniform(0, 2, 99))]
    q, j = np.array([45, 70]), np.array([2, 35])
    plan = context_plan(path, q, j)
    tensor = {k: torch.from_numpy(v) for k, v in plan.items()}
    result = context_score(torch.from_numpy(x), tensor).numpy()
    c = x[:, 288:] / np.linalg.norm(x[:, 288:], axis=1, keepdims=True)
    qn, jn, valid = neighbor_plans(path, q, j)["arc_unique"]
    center = (c[q] * c[j]).sum(-1)
    pairs = np.zeros(qn.shape)
    pairs[valid] = (c[qn[valid]] * c[jn[valid]]).sum(-1)
    means = path_means(center, pairs, valid, repeated_candidate_weights(jn, valid))
    expected = (4 / 15) * center + (11 / 15) * means.max(0)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)
    for at in range(len(q)):
        end = q[at] + 16
        short = context_plan(path[:end], q[at:at+1], j[at:at+1])
        for name in plan:
            np.testing.assert_array_equal(short[name], plan[name][at:at+1])
    np.testing.assert_allclose(plan["weight"].sum(-1), 1., atol=1e-7)


def test_nonrevisit_gradient_and_positive_only_control():
    logits = torch.tensor([2., 3., -1.], requires_grad=True)
    correct = torch.tensor([1., 0., 0.])
    positive = torch.tensor([True, True, False])
    loss = acceptance_loss(logits, correct, positive, True)
    loss.backward()
    assert logits.grad[0] < 0 and (logits.grad[1:] > 0).all()
    logits.grad.zero_()
    acceptance_loss(logits, correct, positive, False).backward()
    assert logits.grad[2] == 0
    assert logits.grad[0] < 0 and logits.grad[1] > 0
    empty = acceptance_loss(logits, correct, torch.zeros_like(positive), False)
    assert torch.isfinite(empty) and empty == 0


def test_zero_vectors_and_network_gradient():
    plan = context_plan(np.arange(90.), np.array([60]), np.array([25]))
    plan = {k: torch.from_numpy(v) for k, v in plan.items()}
    zeros = torch.zeros(90, 416)
    assert context_score(zeros, plan).item() == 0
    torch.manual_seed(3)
    x = torch.randn(90, 416, requires_grad=True)
    context_score(x, plan).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, 288:].abs().sum() > 0
    assert x.grad[:, :288].abs().sum() == 0
