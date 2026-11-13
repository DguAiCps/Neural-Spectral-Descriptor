"""절대 점수 보존·원시 가중합 내보내기·문맥 기울기를 검사한다."""
import torch

from src.retrieval.candidate_learning import candidate_loss
from src.retrieval.linear_fusion import LinearFusionScorer, fusion_features


def test_absolute_quality_and_candidate_independence():
    model = LinearFusionScorer([0.5] * 4, [0.2] * 4)
    good = torch.tensor([[[0.9] * 4, [0.8] * 4]])
    bad = good - 0.6
    assert model(good).min() > model(bad).max()
    extended = torch.cat([good, torch.full((1, 1, 4), 100.)], dim=1)
    torch.testing.assert_close(model(good), model(extended)[:, :2], rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == 5


def test_export_and_monotonicity():
    torch.manual_seed(5)
    model = LinearFusionScorer([0.4, 0.7, 0.2, -0.1], [0.3, 0.1, 0.4, 0.2])
    features = torch.rand(3, 7, 4)
    exported = model.export()
    direct = (features * torch.tensor(exported["weights"])).sum(-1) + exported["intercept"]
    torch.testing.assert_close(model(features), direct, rtol=0, atol=0)
    assert (torch.tensor(exported["weights"]) > 0).all()
    assert (model(features + 0.1) > model(features)).all()


def test_context_gradient_padding_and_future_invariance():
    torch.manual_seed(3)
    raw = torch.randn(80, 416, requires_grad=True)
    emb = torch.nn.functional.normalize(raw, dim=1)
    q = torch.tensor([50, 51])
    c = torch.tensor([[1, 3, 20], [0, 4, 21]])
    valid = c >= 0
    static = torch.rand(2, 3, 3)
    labels = torch.tensor([[True, True, False], [False, False, False]])
    model = LinearFusionScorer([0.] * 4, [1.] * 4)
    features = fusion_features(emb, q, c, static, valid)
    loss = candidate_loss(model(features), labels, valid)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()
    assert raw.grad[q, 288:].abs().sum() > 0
    assert raw.grad[67:].abs().sum() == 0
    changed = emb.detach().clone()
    changed[67:] = 100.
    torch.testing.assert_close(features, fusion_features(changed, q, c, static, valid))
    padded = torch.cat([c, torch.full((2, 2), -1)], 1)
    fpad = fusion_features(emb.detach(), q, padded,
                          torch.cat([static, torch.full((2, 2, 3), 100.)], 1), padded >= 0)
    ypad = torch.cat([labels, torch.zeros(2, 2, dtype=torch.bool)], 1)
    torch.testing.assert_close(loss.detach(), candidate_loss(model(fpad), ypad, padded >= 0))
