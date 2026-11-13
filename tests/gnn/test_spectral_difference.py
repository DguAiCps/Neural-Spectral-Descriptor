"""복소 차이 전달의 좌표 불변성과 유한 범위를 검증한다."""

import torch

from gnn.spectral_difference import transport_difference


def fixture():
    torch.manual_seed(13)
    n = 31
    z = torch.randn(n, 16, 8, dtype=torch.complex128)
    ids = (torch.arange(n)[:, None] + torch.arange(-5, 6)).clamp(0, n - 1)
    mask = (
        (torch.arange(n)[:, None] + torch.arange(-5, 6) >= 0)
        & (torch.arange(n)[:, None] + torch.arange(-5, 6) < n)
    ).double()
    mask[:, 5] = 0
    phase = torch.exp(1j * torch.randn(n, 1) * torch.arange(1, 9)).to(z.dtype)
    eta = torch.full((2, n, 1), 0.25, dtype=torch.float64)
    return z, ids, mask, phase, eta


def test_identity_and_coordinate_covariance():
    z, ids, mask, phase, eta = fixture()
    unit = z / z.abs().flatten(1).norm(dim=1)[:, None, None]
    got = transport_difference(z, ids, mask, phase, eta * 0)
    torch.testing.assert_close(got, unit, atol=1e-7, rtol=1e-7)
    angle = torch.linspace(-2, 2, len(z), dtype=torch.float64)[:, None]
    rotation = torch.exp(1j * angle * torch.arange(1, 9))
    expected = transport_difference(z, ids, mask, phase, eta) * rotation[:, None]
    changed = transport_difference(
        z * rotation[:, None], ids, mask, phase * rotation, eta
    )
    torch.testing.assert_close(changed, expected, atol=1e-7, rtol=1e-7)


def test_prefix_gradient_and_constant_signal():
    z, ids, mask, phase, eta = fixture()
    eta.requires_grad_()
    got = transport_difference(z, ids, mask, phase, eta)
    changed = z.clone()
    changed[21:] *= 100j
    other = transport_difference(changed, ids, mask, phase, eta)
    torch.testing.assert_close(got[:11], other[:11], atol=0, rtol=0)
    got.abs().sum().backward()
    assert torch.isfinite(eta.grad).all() and eta.grad.abs().sum() > 0
    constant = z[:1].expand_as(z) * phase[:, None]
    actual = transport_difference(constant, ids, mask, phase, eta)
    expected = constant / constant.abs().flatten(1).norm(dim=1)[:, None, None]
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    zero = transport_difference(z * 0, ids, mask * 0, phase, eta)
    assert torch.isfinite(zero).all() and not zero.any()
