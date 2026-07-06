"""OFDM transmitter processing."""

from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from src.transmitter.resource_grid import (
    map_data_and_pilots,
    map_data_to_subcarriers,
)


def add_cyclic_prefix(
    time_signal: NDArray[np.complex128],
    cp_length: int,
) -> NDArray[np.complex128]:
    """Copy the end of an OFDM symbol to its beginning."""
    signal = np.asarray(time_signal, dtype=np.complex128)

    if signal.ndim != 1:
        raise ValueError("Time-domain signal must be one-dimensional.")

    if cp_length < 0:
        raise ValueError("Cyclic-prefix length cannot be negative.")

    if cp_length > len(signal):
        raise ValueError(
            "Cyclic-prefix length cannot exceed the OFDM symbol length."
        )

    if cp_length == 0:
        return signal.copy()

    return np.concatenate((signal[-cp_length:], signal))


def ofdm_modulate(
    symbols: Iterable[complex],
    fft_size: int = 64,
    cp_length: int = 16,
) -> NDArray[np.complex128]:
    """Create an OFDM waveform without reserved pilot subcarriers."""
    resource_grid = map_data_to_subcarriers(
        symbols,
        fft_size,
    )

    time_signal = np.fft.ifft(resource_grid)

    return add_cyclic_prefix(time_signal, cp_length)


def ofdm_modulate_with_pilots(
    symbols: Iterable[complex],
    fft_size: int = 64,
    cp_length: int = 16,
    pilot_indices: Sequence[int] = (7, 21, 43, 57),
    pilot_symbol: complex = 1 + 0j,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.int64],
]:
    """
    Create an OFDM waveform containing known pilot symbols.

    Returns:
        Time-domain OFDM waveform.
        Resource-grid indices containing data symbols.
    """
    resource_grid, data_indices = map_data_and_pilots(
        symbols=symbols,
        fft_size=fft_size,
        pilot_indices=pilot_indices,
        pilot_symbol=pilot_symbol,
    )

    time_signal = np.fft.ifft(resource_grid)
    transmitted_signal = add_cyclic_prefix(
        time_signal,
        cp_length,
    )

    return transmitted_signal, data_indices
