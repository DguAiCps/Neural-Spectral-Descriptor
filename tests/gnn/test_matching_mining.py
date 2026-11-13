"""오답 교체가 정답·학습량·허용 후보를 바꾸지 않는지 검사한다."""
import numpy as np

from gnn.matching_mining import paired_triplets


def test_paired_triplets_preserve_positives_and_budget():
    positives = [np.array([1, 2]), np.array([0]), np.array([], dtype=int)]
    old = np.array([7, 8, 9])
    pool = np.array([[10, 11], [-1, -1], [-1, -1]])
    baseline, changed = paired_triplets(positives, old, pool, seed=41, count=4)
    assert baseline.shape == changed.shape == (8, 3)
    assert baseline.dtype == changed.dtype == np.int64
    np.testing.assert_array_equal(baseline[:, :2], changed[:, :2])
    assert (baseline[:, 2] != changed[:, 2]).sum() == 2
    assert np.all(changed[changed[:, 0] == 1, 2] == 8)
    assert set(changed[changed[:, 0] == 0, 2]) <= {7, 10, 11}
    assert np.isfinite(changed).all()
    np.testing.assert_array_equal(changed, paired_triplets(positives, old, pool, 41, count=4)[1])


def test_empty_pool_and_disabled_fraction_keep_reference():
    positives = [np.array([1]), np.array([0])]
    old = np.array([8, -1])
    for pool, fraction in ((np.full((2, 2), -1), 0.5), (np.full((2, 2), 10), 0.0)):
        baseline, changed = paired_triplets(positives, old, pool, 3, fraction)
        np.testing.assert_array_equal(baseline, changed)
        assert len(baseline) == 3
