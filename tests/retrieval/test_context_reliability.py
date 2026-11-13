"""문맥 신뢰도 연결의 보존 성질과 미분을 검사한다."""

import torch

from retrieval.context_reliability import reliability_score


def fixture():
    torch.manual_seed(17)
    raw = torch.nn.functional.normalize(torch.randn(8, 3, dtype=torch.float64), dim=-1)
    ctx = 0.5 * torch.nn.functional.normalize(
        torch.randn(8, 4, dtype=torch.float64), dim=-1
    )
    embedding = torch.cat([raw, ctx], -1).requires_grad_()
    plan = dict(
        query=torch.tensor([[[0, 1, 2], [0, 1, 2]]]),
        candidate=torch.tensor([[[4, 5, 6], [4, 6, 5]]]),
        weight=torch.tensor(
            [[[0.5, 0.3, 0.2], [0.5, 0.25, 0.25]]], dtype=torch.float64
        ),
    )
    peaks = torch.rand(1, 2, 3, 4, dtype=torch.float64)
    peaks[..., 0] /= 30
    peaks[..., 2] /= 30
    return embedding, plan, peaks


def test_zero_matches_original_and_positive_peaks_are_attenuated():
    key, plan, peaks = fixture()
    channel = (plan["weight"][..., None] * peaks).sum(-2).max(1).values
    expected = 0.5 * (
        30 * (0.75 * channel[:, 0] + 0.25 * channel[:, 2])
        + 0.75 * channel[:, 1]
        + 0.25 * channel[:, 3]
    )
    for mode in ("attenuate", "reweight"):
        score, mass = reliability_score(key, plan, peaks, 0, mode, 3)
        torch.testing.assert_close(score, expected, rtol=0, atol=1e-15)
        torch.testing.assert_close(mass, torch.ones_like(mass))
    score, mass = reliability_score(key, plan, peaks, 4, "attenuate", 3)
    assert torch.all(score <= expected) and torch.all(mass < 1)


def test_reweight_preserves_constant_evidence_and_ignores_invalid_pairs():
    key, plan, peaks = fixture()
    peaks[:] = torch.tensor([0.01, 0.3, 0.02, 0.4], dtype=peaks.dtype)
    score, _ = reliability_score(key, plan, peaks, 4, "reweight", 3)
    torch.testing.assert_close(score, torch.tensor([0.35], dtype=score.dtype))
    plan["weight"][..., -1] = 0
    first, _ = reliability_score(key, plan, peaks, 1, "attenuate", 3)
    peaks[..., -1, :] = 999
    second, _ = reliability_score(key, plan, peaks, 1, "attenuate", 3)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_finite_gradient_and_shared_rotation_invariance():
    key, plan, peaks = fixture()
    for mode in ("attenuate", "reweight"):
        score, _ = reliability_score(key, plan, peaks, 1, mode, 3)
        (gradient,) = torch.autograd.grad(score.sum(), key, retain_graph=True)
        assert torch.isfinite(gradient).all() and gradient[:, 3:].abs().sum() > 0
        transform, _ = torch.linalg.qr(torch.randn(4, 4, dtype=key.dtype))
        rotated = torch.cat([key[:, :3], key[:, 3:] @ transform], -1)
        other, _ = reliability_score(rotated, plan, peaks, 1, mode, 3)
        torch.testing.assert_close(score, other, atol=1e-14, rtol=0)
