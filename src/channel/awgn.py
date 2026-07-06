"""Additive white Gaussian noise channel."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def add_awgn(
    signal: ArrayLike,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """
    Add complex Gaussian noise at a specified signal-to-noise ratio.

    Args:
        signal: Complex time-domain samples.
        snr_db: Signal-to-noise ratio in decibels.
        rng: Optional NumPy random-number generator.
    """
    signal_array = np.asarray(signal, dtype=np.complex128)

    if signal_array.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal_array.size == 0:
        return signal_array.copy()

    signal_power = np.mean(np.abs(signal_array) ** 2)

    if signal_power <= 0:
        raise ValueError("Signal power must be greater than zero.")

    if snr_db == np.inf:
        return signal_array.copy()

    if not np.isfinite(snr_db):
        raise ValueError("SNR must be finite or positive infinity.")

    generator = rng if rng is not None else np.random.default_rng()

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)

    noise = noise_std * (
        generator.standard_normal(signal_array.size)
        + 1j * generator.standard_normal(signal_array.size)
    )

    return signal_array + noise
