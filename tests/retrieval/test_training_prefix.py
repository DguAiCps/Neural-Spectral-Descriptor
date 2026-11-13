"""학습 모집단 진단의 과거 거리 계산을 작은 전수 계산과 비교한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from _pdv2_training_coverage import prefix_distance


def test_blocked_prefix_matches_brute_force_on_revisit_and_boundaries():
    rng = np.random.default_rng(10)
    positions = np.cumsum(rng.normal(size=(113,3)),axis=0)
    positions[70:80] = positions[10:20]
    expected = np.r_[np.full(30,np.nan), [np.linalg.norm(positions[:q-29]-positions[q],axis=1).min()
                                        for q in range(30,len(positions))]]
    for block in (1,17,256):
        np.testing.assert_allclose(prefix_distance(positions,block), expected, atol=1e-12,rtol=0)
