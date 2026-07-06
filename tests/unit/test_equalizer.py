import numpy as np
import pytest

from src.receiver.equalizer import equalize_flat_channel


def test_equalizer_recovers_original_symbols():
    original = np.array(
        [1 + 1j, -1 + 1j, -1 - 1j],
        dtype=np.complex128,
    )

    coefficient = 0.4 + 0.7j
    received = original * coefficient

    equalized = equalize_flat_channel(
        received,
        coefficient,
    )

    np.testing.assert_allclose(
        equalized,
        original,
        atol=1e-12,
    )


def test_equalizer_rejects_zero_channel():
    with pytest.raises(ValueError):
        equalize_flat_channel(
            [1 + 1j],
            channel_coefficient=0,
        )
