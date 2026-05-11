import torch

from encoding.cross_spectrum import (
    adjacent_cross_spectrum_dim,
    normalized_adjacent_cross_spectrum,
)


def test_adjacent_cross_spectrum_dim():
    assert adjacent_cross_spectrum_dim(16, 8) == 15 * 8 * 2
    assert adjacent_cross_spectrum_dim(1, 8) == 0
    assert adjacent_cross_spectrum_dim(16, 0) == 0


def test_adjacent_cross_spectrum_is_integer_shift_invariant():
    generator = torch.Generator().manual_seed(7)
    image = torch.randn((16, 360), generator=generator)
    shifted = torch.roll(image, shifts=37, dims=1)

    fft = torch.fft.rfft(image, dim=1, norm="ortho")
    fft_shifted = torch.fft.rfft(shifted, dim=1, norm="ortho")

    feat = normalized_adjacent_cross_spectrum(fft, n_freqs=8)
    feat_shifted = normalized_adjacent_cross_spectrum(fft_shifted, n_freqs=8)

    assert torch.max(torch.abs(feat - feat_shifted)).item() < 1e-5


def test_adjacent_cross_spectrum_preserves_relative_phase():
    image = torch.zeros((2, 64))
    t = torch.arange(64, dtype=torch.float32)
    image[0] = torch.cos(2 * torch.pi * t / 64)
    image[1] = torch.cos(2 * torch.pi * t / 64 + torch.pi / 3)

    fft = torch.fft.rfft(image, dim=1, norm="ortho")
    feat = normalized_adjacent_cross_spectrum(fft, n_freqs=1)

    assert feat.numel() == 2
    assert abs(float(torch.linalg.norm(feat)) - 1.0) < 1e-5
    assert abs(float(feat[1])) > 0.5
