"""학습 계산 캐시가 실제 삼중항과 난수 소비를 바꾸지 않는지 검사한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _train_lag15_cached_mining import _CACHE, cached_worker, install
from gnn import triplet_miner


def test_cached_triplets_and_rng_equal_original():
    """양성 재추출·최난도 오답·난수 상태가 원래 작업 함수와 같아야 한다."""
    install()
    rng = np.random.default_rng(123)
    n = 96
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    poses[:, 0, 3] = np.tile(np.arange(24), 4)
    descriptors = rng.normal(size=(n, 12)).astype(np.float32)
    descriptors[80] = descriptors[79]  # 동률 오답이 있어도 첫 후보를 고른다.
    args = (2, np.arange(n) + 500, descriptors, poses, 1., 20, 5., 20, "hard", 3, "l2")
    _CACHE.clear()
    for seed in (5, 9):
        np.random.seed(seed)
        expected = triplet_miner._original_lag15_worker(args)
        expected_random = np.random.random(8)
        np.random.seed(seed)
        actual = cached_worker(args)
        actual_random = np.random.random(8)
        assert expected == actual and len(actual[1]) > 0
        np.testing.assert_array_equal(expected_random, actual_random)
    assert len(_CACHE) == 1
    descriptors[0, 0] += 10
    cached_worker(args)
    assert len(_CACHE) == 2


def test_no_positive_and_nonhard_fallback():
    """양성 없음과 무작위 오답 모드도 원래 결과와 같아야 한다."""
    install()
    n = 50
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    poses[:, 0, 3] = np.arange(n) * 10
    descriptors = np.ones((n, 8), np.float32)
    args = (3, np.arange(n), descriptors, poses, 1., 20, 5., 20, "hard", 3, "l2")
    assert cached_worker(args) == triplet_miner._original_lag15_worker(args)
    poses[:, 0, 3] = np.tile(np.arange(25), 2)
    args = (*args[:8], "random", *args[9:])
    np.random.seed(4)
    expected = triplet_miner._original_lag15_worker(args)
    np.random.seed(4)
    assert cached_worker(args) == expected
