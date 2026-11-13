"""점수 상한·연속 경로 벌점의 불변성과 입력 검사를 확인한다."""
import numpy as np
import pytest

from retrieval.path_confidence import confidence_variants


def test_cap_and_soft_penalty_keep_candidates_and_finite_scores():
    old = np.array([.9, .6, .4, 0.])
    new = np.array([.95, .5, .7, .3])
    margin = np.array([.8, 0., -.2, -.7])
    result = confidence_variants(old, new, margin)
    np.testing.assert_array_equal(result["capped"], [.9, .5, .4, 0.])
    np.testing.assert_allclose(result["soft_null"], new*np.exp([0., 0., -.2, -.7]))
    assert (result["soft_capped"] <= old).all()
    assert (result["soft_capped"] <= new).all()
    assert all(x.shape == old.shape and x.dtype == np.float64 and np.isfinite(x).all()
               for x in result.values())


def test_continuous_at_null_boundary_and_score_scale_equivariance():
    old, new = np.array([.5, .5, .5]), np.array([.6, .6, .6])
    margin = np.array([-1e-8, 0., 1e-8])
    result = confidence_variants(old, new, margin)
    assert abs(result["soft_capped"][0]-result["soft_capped"][1]) < 1e-8
    scaled = confidence_variants(old/30, new/30, margin)
    for key in result:
        np.testing.assert_allclose(scaled[key]*30, result[key], atol=1e-15)


def test_prefix_and_pointwise_independence():
    rng = np.random.default_rng(114)
    old, new, margin = rng.random(100), rng.random(100), rng.uniform(-.8, .8, 100)
    full = confidence_variants(old, new, margin)
    short = confidence_variants(old[:60], new[:60], margin[:60])
    for key in full:
        np.testing.assert_array_equal(full[key][:60], short[key])
    assert np.isfinite(full["soft_capped"]).all()


def test_invalid_inputs_are_not_silent_rejections():
    for args in (([1.], [np.nan], [0.]), ([1.], [1., 2.], [0.]), ([-1.], [1.], [0.])):
        with pytest.raises(ValueError):
            confidence_variants(*args)
