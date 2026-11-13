"""거리 대응의 속도 차이와 관측 범위를 합성 궤적으로 검사한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _pdv2_arc_evidence import arc_neighbor


def test_arc_neighbors_match_distance_and_stay_in_observed_window():
    path = np.arange(60)*2.
    centers = np.array([0, 15, 40, 59])
    target = np.array([-8., 6., -6., 8.])
    actual = arc_neighbor(path, centers, target)
    np.testing.assert_array_equal(actual, [0, 18, 37, 59])
    assert np.all(np.abs(actual-centers)<=10)
    slow = np.arange(60)*.5
    np.testing.assert_array_equal(arc_neighbor(slow, np.array([20]), np.array([3.])), [26])
