"""Flat Rayleigh fading channel models."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def apply_flat_rayleigh_fading(
    signal: ArrayLike,
    coefficient: complex | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex128], complex]:
    """
    Apply one complex fading coefficient to an entire waveform.

    When no coefficient is supplied, a normalized Rayleigh fading
    coefficient is randomly generated.
    """
    signal_array = np.asarray(signal, dtype=np.complex128)

    if signal_array.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if coefficient is None:
        generator = rng if rng is not None else np.random.default_rng()

        coefficient = complex(
            (
                generator.standard_normal()
                + 1j * generator.standard_normal()
            )
            / np.sqrt(2)
        )
    else:
        coefficient = complex(coefficient)

    if not (
        np.isfinite(coefficient.real)
        and np.isfinite(coefficient.imag)
    ):
        raise ValueError("Channel coefficient must be finite.")

    faded_signal = signal_array * coefficient

    return faded_signal, coefficient
