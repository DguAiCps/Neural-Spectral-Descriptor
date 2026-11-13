"""후보의 시간·거리 제한과 코사인 진단을 독립 예제로 검산한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _center_information_lib import fixed_triplets, triplet_statistics


def test_past_positive_and_hard_negative_exclude_near_and_future():
    xyz = np.array([[0., 0, 0], [40., 0, 0], [15., 0, 0], [50., 0, 0],
                    [0., 0, 0], [80., 0, 0]])
    raw = np.array([[0., 1], [.8, .6], [1., 0], [.9, .1], [1., 0], [1., 0]])
    actual = fixed_triplets(raw, xyz, np.array([4]), exclusion=2)
    # 3은 더 비슷해도 시간 제한, 2는 회색 거리, 5는 미래이므로 제외한다.
    np.testing.assert_array_equal(actual, [[4, 0, 1]])


def test_nearest_positive_tie_and_three_dimensional_negative():
    xyz = np.array([[0., 0, 0], [0., 0, 0], [0., 0, 30], [0., 0, 0]])
    raw = np.ones((4, 2))
    np.testing.assert_array_equal(fixed_triplets(raw, xyz, np.array([3]), 1), [[3, 0, 2]])
    assert fixed_triplets(raw, xyz, np.array([0]), 1).shape == (0, 3)


def test_cosine_margin_dense_reference_and_zero_vector():
    rng = np.random.default_rng(82)
    features = rng.normal(size=(12, 8)).astype(np.float32)
    features[11] = 0
    triplets = np.array([[0, 1, 2], [4, 3, 8], [10, 11, 0]])
    result = triplet_statistics(features, triplets)
    for i, (q, p, n) in enumerate(triplets):
        def cosine(a, b):
            a, b = features[a].astype(float), features[b].astype(float)
            return a.dot(b) / max(np.linalg.norm(a), 1e-12) / max(np.linalg.norm(b), 1e-12)
        np.testing.assert_allclose(result["margin"][i], cosine(q, p)-cosine(q, n), atol=1e-14)
    np.testing.assert_array_equal(result["valid"], [True, True, False])
    assert all(np.isfinite(v).all() for v in result.values())
