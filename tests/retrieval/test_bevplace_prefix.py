"""GPU 제안 뒤 공식 CPU 재계산이 접두 검색과 동점을 보존하는지 검사한다."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from _bevplace_official_dump import ledger_l2
from _bevplace_prefix_exact import ledger_exact


def test_gpu_proposal_matches_full_cpu_for_duplicates_and_expansion():
    rng = np.random.default_rng(31)
    vectors = rng.normal(size=(97, 8192)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors[1] = vectors[0]
    vectors[30] = vectors[0]
    positions = rng.normal(size=(97, 3))
    tied = np.repeat(vectors[:1], len(vectors), axis=0)
    for descriptors in (vectors, tied):
        actual, audit = ledger_exact(descriptors, positions)
        expected = ledger_l2(descriptors, positions)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
            assert actual[key].dtype == expected[key].dtype
            assert np.isfinite(actual[key]).all()
        assert (actual["j_star"] <= actual["query_idx"] - 30).all()
        assert audit["all_candidates_past_only"]
        if descriptors is tied:
            assert audit["shortlist_max"] > 32
            np.testing.assert_array_equal(actual["j_star"], 0)
