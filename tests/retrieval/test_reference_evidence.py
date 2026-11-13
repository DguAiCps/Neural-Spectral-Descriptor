"""배경 참조의 과거 범위와 보정의 수치 성질을 검사한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _pdv2_reference_evidence import reference_indices, calibrate


def test_reference_sampling_is_unique_and_past_only():
    q = np.array([30, 31, 70, 200, 20000])
    refs, valid = reference_indices(q)
    assert refs.shape == valid.shape == (5, 64) and refs.dtype == np.int64
    for query, row, mask in zip(q, refs, valid):
        chosen = row[mask]
        assert len(chosen) == min(64, query-29)
        assert len(np.unique(chosen)) == len(chosen)
        assert np.all((chosen >= 0) & (chosen <= query-30))


def test_background_z_affine_invariance_and_fallback():
    refs, valid = reference_indices(np.array([200, 30]))
    scores = np.broadcast_to(np.linspace(.1, .7, 64), refs.shape).copy()
    raw, temporal, winner = np.array([.8, .5]), np.array([.6, .7]), np.array([90, 0])
    a = calibrate(raw, temporal, refs, valid, scores, winner)
    b = calibrate(3*raw+2, 3*temporal+2, refs, valid, 3*scores+2, winner)
    np.testing.assert_allclose(a["reference_z"][0], b["reference_z"][0], atol=1e-12)
    assert not a["fallback"][0] and a["fallback"][1]
    assert a["reference_z"][1] == raw[1]
    np.testing.assert_allclose(a["temporal_blend25"], [.75, .55])
    assert all(np.isfinite(v).all() for v in a.values())
