"""융합의 원래 연산 재현과 후보 범위에 따른 순위 변화를 검산한다."""
import numpy as np

from _pdv2_fusion_factor_lib import fuse_with_range, pair_components


def test_original_float32_formula_and_constant_channel():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(17, 4)).astype(np.float32)
    values[:, 3] = .25
    weights = (1., 1., 2., 1.)
    expected = sum(w * (values[:, i] - float(values[:, i].min())) /
                   max(float(values[:, i].max()) - float(values[:, i].min()), 1e-8)
                   for i, w in enumerate(weights))
    actual, ranges = fuse_with_range(values, weights)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (17,) and actual.dtype == np.float32
    assert np.isfinite(actual).all() and ranges[3, 2] == 1e-8


def test_losing_distractor_changes_pair_order_through_normalization():
    values = np.array([[.9, .4, 0, 0], [.8, .6, 0, 0], [0, 0, 0, 0],
                       [0, 10, 0, 0]], dtype=np.float32)
    weights = (1., 1., 0., 0.)
    old, old_range = fuse_with_range(values[:3], weights)
    new, new_range = fuse_with_range(values, weights)
    assert old.argmax() == 1 and new.argmax() == 0
    assert new[3] < new[0]  # 추가한 후보 자신은 선택되지 않는다.
    only_range, _ = fuse_with_range(values[:3], weights, values)
    assert only_range.argmax() == 0
    components = pair_components(values[0], values[1], new_range, weights)
    np.testing.assert_allclose(components.sum(), new[0] - new[1], atol=1e-7)
    assert pair_components(values[0], values[1], old_range, weights).sum() < 0


def test_fixed_reference_preserves_pair_ranking_and_permutation():
    rng = np.random.default_rng(13)
    values = rng.normal(size=(12, 4)).astype(np.float32)
    weights = (1., 1., 2., 1.)
    ref = values[:6]
    a, _ = fuse_with_range(values[:6], weights, ref)
    b, _ = fuse_with_range(values, weights, ref)
    np.testing.assert_array_equal(a, b[:6])
    perm = rng.permutation(len(values))
    c, _ = fuse_with_range(values[perm], weights, ref)
    np.testing.assert_array_equal(b[perm], c)
