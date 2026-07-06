import numpy as np

from src.channel.fading import apply_flat_rayleigh_fading


def test_known_fading_coefficient():
    signal = np.array(
        [1 + 0j, 2 + 1j],
        dtype=np.complex128,
    )

    coefficient = 0.5 + 0.25j

    faded, returned_coefficient = apply_flat_rayleigh_fading(
        signal,
        coefficient=coefficient,
    )

    np.testing.assert_allclose(
        faded,
        signal * coefficient,
    )

    assert returned_coefficient == coefficient


def test_random_fading_preserves_shape():
    signal = np.ones(64, dtype=np.complex128)

    faded, coefficient = apply_flat_rayleigh_fading(
        signal,
        rng=np.random.default_rng(123),
    )

    assert faded.shape == signal.shape
    assert np.isfinite(coefficient.real)
    assert np.isfinite(coefficient.imag)
