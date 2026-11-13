"""불가능한 교정 제외와 정규화 후 정답 보호의 실제 동작을 검사한다."""
import torch

from src.retrieval.fusion_guard import fusion_guard_loss, fusion_guard_terms


def test_static_bound_excludes_impossible_query_without_claiming_sufficiency():
    key = torch.tensor([[.7, .6, .2], [.7, .6, .2]], requires_grad=True)
    static = torch.tensor([[[0., 0., 0.], [1., 1., 1.], [.5, .5, .5]],
                           [[.7, .7, .7], [.6, .6, .6], [.2, .2, .2]]])
    valid = torch.ones_like(key, dtype=torch.bool)
    correct = torch.tensor([[True, False, False], [True, False, False]])
    terms = fusion_guard_terms(key, key.detach(), static, valid, correct, (1., 1., 2., 1.))
    assert terms['admissible'].tolist() == [False, True]
    assert terms['static_upper_bound'][0] < 0
    fusion_guard_loss(terms, True, False).backward()
    assert key.grad[0].abs().sum() == 0 and torch.isfinite(key.grad).all()


def test_guard_is_one_sided_and_does_not_copy_teacher_mistakes():
    teacher = torch.tensor([[.7, .6, .2], [.6, .7, .2]])
    key = torch.tensor([[.6, .7, .2], [.6, .7, .2]], requires_grad=True)
    valid = torch.ones_like(key, dtype=torch.bool)
    correct = torch.tensor([[True, False, False], [True, False, False]])
    static = torch.zeros(2, 3, 3)
    terms = fusion_guard_terms(key, teacher, static, valid, correct, (1., 1., 2., 1.))
    assert terms['protected'].tolist() == [True, False]
    assert terms['guard'][0] > 0 and terms['guard'][1] == 0
    terms['guard'].mean().backward()
    assert key.grad[0, 0] < 0 and key.grad[0, 1] > 0
    assert key.grad[1].abs().sum() == 0
    same = fusion_guard_terms(teacher, teacher, static, valid, correct, (1., 1., 2., 1.))
    assert same['guard'].sum() == 0


def test_guard_detects_damage_from_third_candidate_range_without_pair_reversal():
    teacher = torch.tensor([[.8, .7, .3, .5]])
    current = torch.tensor([[.8, .7, -.7, .5]], requires_grad=True)
    static = torch.tensor([[[.875, .875, .875], [.9125, .9125, .9125],
                            [0., 0., 0.], [1., 1., 1.]]])
    correct = torch.tensor([[True, False, False, False]])
    valid = torch.ones_like(correct)
    terms = fusion_guard_terms(current, teacher, static, valid, correct, (1., 1., 2., 1.))
    assert current[0, 0] > current[0, 1]
    assert terms['reference_margin'][0] > 0 and terms['margin'][0] < 0
    terms['guard'].sum().backward()
    assert torch.isfinite(current.grad).all() and current.grad[0, 2] < 0


def test_invalid_labels_constant_channels_and_padding_have_finite_zero_signal():
    key = torch.full((2, 4), .5, requires_grad=True)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    correct = torch.tensor([[False, False, False, False], [True, True, False, False]])
    terms = fusion_guard_terms(key, key.detach(), torch.zeros(2, 4, 3), valid,
                               correct, (1., 1., 2., 1.))
    loss = fusion_guard_loss(terms, True, True)
    loss.backward()
    assert loss == 0 and key.grad.abs().sum() == 0
    assert all(torch.isfinite(v).all() for v in terms.values())


def test_padding_and_positive_affine_key_transform_preserve_loss():
    torch.manual_seed(74)
    key = torch.rand(3, 8)
    static = torch.rand(3, 8, 3)
    correct = torch.zeros(3, 8, dtype=torch.bool)
    correct[:, :2] = True
    valid = torch.ones_like(correct)
    before = fusion_guard_terms(key, key, static, valid, correct, (1., 1., 2., 1.))
    padded_key = torch.cat((2 * key + 3, torch.full((3, 2), 100.)), 1)
    padded_static = torch.cat((static, torch.full((3, 2, 3), 100.)), 1)
    pad = torch.zeros(3, 2, dtype=torch.bool)
    after = fusion_guard_terms(padded_key, padded_key, padded_static,
        torch.cat((valid, pad), 1), torch.cat((correct, pad), 1), (1., 1., 2., 1.))
    for name in before:
        torch.testing.assert_close(before[name], after[name], atol=2e-5, rtol=2e-5)
