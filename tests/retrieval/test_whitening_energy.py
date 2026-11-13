"""에너지 보존 대조의 기대 에너지와 회전 성질을 검사한다."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]
import _pdv2_whitening_energy as run


def test_energy_preservation_and_yaw():
    """학습 기대값·스캔별 세기·영벡터·회전 위상 교환을 확인한다."""
    rng = np.random.default_rng(91)
    x = (rng.normal(size=(96, 16, 8)) + 1j * rng.normal(size=(96, 16, 8))).astype(
        np.complex64
    )
    x[:, :, 4:] *= 0.2
    x[0] = 0
    z = x.astype(np.complex128)
    moment = np.einsum("nik,njk->kij", z, z.conj()) / len(z)
    full = run.source.whitening_matrix(moment, True)
    matrix = run.population_matrix(moment, full)
    np.testing.assert_allclose(
        np.trace(matrix @ moment @ matrix.conj().transpose(0, 2, 1), axis1=1, axis2=2),
        np.trace(moment, axis1=1, axis2=2),
        atol=1e-12,
        rtol=1e-12,
    )
    phase = np.exp(1j * 0.23 * np.arange(1, 9))[None, None]
    for mode, w in (("population", matrix), ("scan", full)):
        y = run.transform(x, w, mode)
        assert y.shape == x.shape and y.dtype == np.complex64
        assert np.isfinite(y).all()
        np.testing.assert_array_equal(y[0], 0)
        np.testing.assert_allclose(
            run.transform(x * phase, w, mode), y * phase, atol=1e-6, rtol=3e-6
        )
        if mode == "scan":
            np.testing.assert_allclose(
                (abs(y.astype(np.complex128)) ** 2).sum(1),
                (abs(z) ** 2).sum(1),
                atol=1e-6,
                rtol=3e-7,
            )
