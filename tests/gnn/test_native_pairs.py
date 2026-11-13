"""새 스캔 표본의 시퀀스·거리 경계와 비복원 예산을 검사한다."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _pdv2_native_pairs_lib import nested_rows, positive_pools


def test_positive_pools_respect_physical_distance_time_and_sequence():
    xyz = np.zeros((90, 3))
    xyz[:, 0] = 100 + np.arange(90)
    xyz[0, 0], xyz[29, 0], xyz[30, 0], xyz[31, 0], xyz[60, 0] = 0, 1, 5, 5.0001, 0
    seq = np.r_[np.zeros(60, np.int64), np.ones(30, np.int64)]
    pools = positive_pools(xyz, seq)
    np.testing.assert_array_equal(pools[0], [30])
    np.testing.assert_array_equal(pools[30], [0])
    for a, p in enumerate(pools):
        assert p.dtype == np.int64
        assert np.all(seq[p] == seq[a]) and np.all(abs(p-a) >= 30)
        assert np.all(np.linalg.norm(xyz[p]-xyz[a], axis=1) <= 5)


def test_budget_selection_is_deterministic_nested_and_without_replacement():
    first = nested_rows(101, 83, 61, 123)
    second = nested_rows(101, 83, 61, 123)
    for k, count in (("large", 83), ("small", 61)):
        assert len(np.unique(first[k])) == count
        np.testing.assert_array_equal(first[k], second[k])
        assert first[k].min() >= 0 and first[k].max() < 101
    assert np.isin(first["small"], first["large"]).all()
    with pytest.raises(ValueError):
        nested_rows(100, 101, 60, 1)
