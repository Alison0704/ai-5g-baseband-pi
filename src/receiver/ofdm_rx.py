"""OFDM receiver processing."""

from typing import Iterable

import numpy as np
from numpy.typing import NDArray


def remove_cyclic_prefix(
    received_signal: Iterable[complex],
    fft_size: int,
    cp_length: int,
) -> NDArray[np.complex128]:
    """Remove the cyclic prefix from one OFDM symbol."""
    signal = np.asarray(list(received_signal), dtype=np.complex128)

    if signal.ndim != 1:
        raise ValueError("Received signal must be one-dimensional.")

    if fft_size <= 0:
        raise ValueError("FFT size must be positive.")

    if cp_length < 0:
        raise ValueError("Cyclic-prefix length cannot be negative.")

    expected_length = fft_size + cp_length

    if len(signal) != expected_length:
        raise ValueError(
            f"Expected {expected_length} samples, received {len(signal)}."
        )

    return signal[cp_length:]


def ofdm_demodulate(
    received_signal: Iterable[complex],
    fft_size: int = 64,
    cp_length: int = 16,
    num_data_symbols: int | None = None,
) -> NDArray[np.complex128]:
    """
    Recover frequency-domain symbols from an OFDM waveform.

    Processing:
        received waveform
        -> remove cyclic prefix
        -> FFT
        -> extract active data subcarriers
    """
    time_signal = remove_cyclic_prefix(
        received_signal,
        fft_size,
        cp_length,
    )

    resource_grid = np.fft.fft(time_signal)

    if num_data_symbols is None:
        return resource_grid

    if not 0 <= num_data_symbols <= fft_size:
        raise ValueError(
            "Number of data symbols must be between 0 and FFT size."
        )

    return resource_grid[:num_data_symbols]
