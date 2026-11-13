"""문맥 이득의 항등·위상 보존·정렬 수식과 미분을 확인한다."""

import numpy as np
import torch

from retrieval.context_spectral_gain import (
    gain_score,
    pair_peaks,
    spectral_bank,
    spectral_gain,
)


def test_identity_rotation_and_bounds():
    torch.manual_seed(7)
    key = torch.randn(5, 416, dtype=torch.float64)
    key[:, :288] = torch.nn.functional.normalize(key[:, :288], dim=-1)
    key[:, 288:] = 0.5 * torch.nn.functional.normalize(key[:, 288:], dim=-1)
    assert torch.equal(spectral_gain(key, 0), torch.ones(5, 16, 8))
    gain = spectral_gain(key, 0.5)
    assert gain.min() >= np.exp(-0.5) and gain.max() <= np.exp(0.5)
    z = torch.randn(5, 16, 8, dtype=torch.complex128)
    phase = torch.exp(1j * torch.arange(1, 9) * 0.3)
    torch.testing.assert_close((z * phase) * gain, (z * gain) * phase)


def test_pair_formula_against_numpy():
    torch.manual_seed(13)
    z = torch.randn(4, 16, 8, dtype=torch.complex128)
    q, j = torch.tensor([0, 1]), torch.tensor([2, 3])
    actual = pair_peaks(spectral_bank(z), q, j).numpy()
    fft = np.pad(z.numpy(), ((0, 0), (0, 0), (1, 22)))
    norm = np.linalg.norm(z.numpy().reshape(4, -1), axis=1)
    spec = np.fft.irfft(fft[j] * fft[q].conj(), n=60).sum(1).max(1) / (
        norm[q] * norm[j]
    )
    grid = np.fft.irfft(fft, n=60)
    grid /= np.linalg.norm(grid, axis=1, keepdims=True)
    cf = np.fft.rfft(grid)
    col = np.fft.irfft(cf[j] * cf[q].conj(), n=60).sum(1).max(1) / 60
    np.testing.assert_allclose(actual, np.stack((spec, col), -1), atol=1e-12, rtol=0)


def test_score_gradients_and_zeros():
    torch.manual_seed(3)
    key = torch.randn(10, 416, dtype=torch.float64, requires_grad=True)
    z = torch.randn(10, 2, 16, 8, dtype=torch.complex128)
    q = torch.tensor([[[0, 1], [0, 2]], [[3, 4], [3, 5]]])
    plan = dict(
        query=q, candidate=q + 4, weight=torch.full_like(q, 0.5, dtype=torch.float64)
    )
    loss = gain_score(key, z, plan, 0.5).sum()
    loss.backward()
    assert torch.isfinite(key.grad).all() and key.grad[:, 288:].norm() > 0
    zero = pair_peaks(
        spectral_bank(torch.zeros_like(z[:, 0])), torch.tensor([0]), torch.tensor([1])
    )
    assert torch.isfinite(zero).all()
