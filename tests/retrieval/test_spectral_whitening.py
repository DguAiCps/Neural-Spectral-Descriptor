"""스펙트럼 통계 보정의 회전 성질과 수치 유한성을 검사한다."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
import _pdv2_spectral_whitening as run


def test_whitening_and_yaw_equivariance():
    """반경 행 보정은 각 주파수의 공통 회전 위상과 교환한다."""
    rng = np.random.default_rng(77)
    x = (rng.normal(size=(48, 16, 8)) + 1j * rng.normal(size=(48, 16, 8))).astype(
        np.complex64
    )
    cov = np.einsum(
        "nik,njk->kij", x.astype(np.complex128), x.conj().astype(np.complex128)
    ) / len(x)
    phase = np.exp(1j * 0.317 * np.arange(1, 9))
    for full in (False, True):
        matrix = run.whitening_matrix(cov, full)
        y = run.transform(x, matrix)
        assert y.shape == x.shape and y.dtype == np.complex64
        assert np.isfinite(y).all()
        np.testing.assert_allclose(
            run.transform(x * phase[None, None], matrix),
            y * phase[None, None],
            atol=5e-7,
            rtol=2e-6,
        )
    identity = np.repeat(np.eye(16)[None], 8, axis=0)
    np.testing.assert_array_equal(run.transform(x, identity), x)
