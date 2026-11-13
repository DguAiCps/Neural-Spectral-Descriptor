"""재방문 차이 적합의 회전 정렬을 합성 입력으로 검산한다."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
import _pdv2_whitening_revisit as run


def test_rotation_alignment():
    """알려진 60구간 회전을 복구하고 차이 모멘트가 양의 준정부호인지 확인한다."""
    rng = np.random.default_rng(182)
    a = rng.normal(size=(40, 16, 8)) + 1j * rng.normal(size=(40, 16, 8))
    a /= np.sqrt(np.sum(abs(a) ** 2, axis=(1, 2)))[:, None, None]
    bins = rng.integers(0, 60, len(a))
    phase = np.exp(-2j * np.pi * bins[:, None] * np.arange(1, 9)[None] / 60)
    b = a * phase[:, None]
    delta, shift, before, after = run.align_pairs(a, b)
    np.testing.assert_array_equal(shift, bins)
    np.testing.assert_allclose(delta, 0, atol=1e-14, rtol=0)
    assert np.max(after) < 1e-25
    b += 0.03 * (rng.normal(size=b.shape) + 1j * rng.normal(size=b.shape))
    delta, shift, before, after = run.align_pairs(a, b)
    assert (after <= before + 1e-12).all()
    moment = np.einsum("nik,njk->kij", delta, delta.conj()) / (2 * len(a))
    assert np.linalg.eigvalsh(moment).min() > -1e-14
