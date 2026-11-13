"""물리적 회전 부호·경계·불변성과 기존 정렬 점수를 검사한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from retrieval.spectral_evidence import correlation_profiles, peak_to_sidelobe, transport_profile, temporal_scores
from _xs_eval_lib import variant_inputs, align_scores, column_align_scores


def test_profiles_reproduce_both_canonical_alignment_scores():
    rng = np.random.default_rng(12)
    sk = (rng.normal(size=(16, 16, 8)) + 1j*rng.normal(size=(16, 16, 8))).astype(np.complex64)
    bank = variant_inputs(sk, 8)
    q, j = np.full(8, 10), np.arange(8)
    for channel, expected in (("c_spec", align_scores(bank[1], bank[2], 10, j)),
                              ("c_col", column_align_scores(bank[3], 10, j))):
        actual = correlation_profiles(bank, q, j, channel)
        np.testing.assert_allclose(actual.max(1), expected, atol=2e-8, rtol=2e-6)
        assert actual.shape == (8, 60) and np.isfinite(actual).all()


def test_motion_compensation_sign_and_reverse_traversal_invariance():
    rng = np.random.default_rng(22)
    world = (rng.normal(size=(16, 8)) + 1j*rng.normal(size=(16, 8))).astype(np.complex64)
    yaw = np.linspace(-.6, 1.2, 25)
    # f_sensor(theta)=f_world(theta+yaw), 따라서 F_sensor=F_world*exp(i*k*yaw).
    sk = world[None] * np.exp(1j*yaw[:, None, None]*np.arange(1, 9)[None, None])
    bank = variant_inputs(sk.astype(np.complex64), 8)
    q, j = np.array([18]), np.array([6])
    center = correlation_profiles(bank, q, j, "c_spec")
    neighbor = correlation_profiles(bank, q+5, j-5, "c_spec")
    delta = (yaw[q+5]-yaw[q]) - (yaw[j-5]-yaw[j])
    np.testing.assert_allclose(transport_profile(neighbor, delta), center, atol=3e-8, rtol=2e-5)
    scores = temporal_scores(bank, yaw, q, j, "c_spec")
    reversed_bank = variant_inputs(sk[::-1].astype(np.complex64), 8)
    reversed_scores = temporal_scores(reversed_bank, yaw[::-1], 24-q, 24-j, "c_spec")
    for name in scores:
        np.testing.assert_allclose(scores[name], reversed_scores[name], atol=1e-10)


def test_psr_flat_and_boundary_support_are_finite():
    np.testing.assert_array_equal(peak_to_sidelobe(np.ones((2, 60))), 0)
    theta = np.arange(60)*2*np.pi/60
    p = np.cos(theta)[None]
    np.testing.assert_allclose(peak_to_sidelobe(p), peak_to_sidelobe(4*p+2), atol=1e-12)
    rng = np.random.default_rng(2)
    sk = (rng.normal(size=(50, 16, 8)) + 1j*rng.normal(size=(50, 16, 8))).astype(np.complex64)
    scores = temporal_scores(variant_inputs(sk, 8), np.zeros(50), np.array([30, 35]), np.array([0, 5]), "c_spec")
    np.testing.assert_array_equal(scores["support_forward"], [2, 3])
    np.testing.assert_array_equal(scores["support_reverse"], [2, 3])
    assert all(np.isfinite(v).all() for v in scores.values())
