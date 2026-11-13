"""절단 키의 계수 순서·차원·회전 불변성을 확인한다."""

import numpy as np
import pytest

from encoding.frequency_key import frequency_key


def test_layout_and_dimension() -> None:
    values = np.arange(2 * 16 * 181, dtype=np.float64).reshape(2, 16, 181)
    for frequencies in (18, 22):
        key = frequency_key(values, frequencies)
        assert key.shape == (2, 16 * frequencies)
        assert key.dtype == np.float32 and np.isfinite(key).all()
        for row in (0, 15):
            np.testing.assert_array_equal(
                key[:, row * frequencies : (row + 1) * frequencies],
                values[:, row, :frequencies],
            )


def test_cyclic_shift_invariance() -> None:
    rng = np.random.default_rng(9)
    image = rng.normal(size=(16, 360))
    a = frequency_key(np.abs(np.fft.rfft(image)), 22)
    b = frequency_key(np.abs(np.fft.rfft(np.roll(image, 37, axis=1))), 22)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        frequency_key(np.zeros((8, 181)), 22)
    with pytest.raises(ValueError):
        frequency_key(np.zeros((16, 12)), 22)
    with pytest.raises(ValueError):
        frequency_key(np.full((16, 181), np.nan), 22)
