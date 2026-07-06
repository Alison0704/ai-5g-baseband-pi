"""OFDM resource-grid mapping functions."""

from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


def map_data_to_subcarriers(
    symbols: Iterable[complex],
    fft_size: int = 64,
) -> NDArray[np.complex128]:
    """Place modulation symbols sequentially into OFDM subcarriers."""
    symbol_array = np.asarray(list(symbols), dtype=np.complex128)

    if symbol_array.ndim != 1:
        raise ValueError("Symbols must be one-dimensional.")

    if fft_size <= 0:
        raise ValueError("FFT size must be positive.")

    if len(symbol_array) > fft_size:
        raise ValueError(
            f"Cannot place {len(symbol_array)} symbols "
            f"into {fft_size} subcarriers."
        )

    grid = np.zeros(fft_size, dtype=np.complex128)
    grid[: len(symbol_array)] = symbol_array

    return grid


def map_data_and_pilots(
    symbols: Iterable[complex],
    fft_size: int = 64,
    pilot_indices: Sequence[int] = (7, 21, 43, 57),
    pilot_symbol: complex = 1 + 0j,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.int64],
]:
    """
    Map data and known pilot symbols into an OFDM resource grid.

    Returns:
        Complete resource grid.
        Indices containing the supplied data symbols.
    """
    symbol_array = np.asarray(list(symbols), dtype=np.complex128)
    pilot_array = np.asarray(pilot_indices, dtype=np.int64)

    if symbol_array.ndim != 1:
        raise ValueError("Symbols must be one-dimensional.")

    if fft_size <= 0:
        raise ValueError("FFT size must be positive.")

    if pilot_array.ndim != 1 or pilot_array.size == 0:
        raise ValueError("At least one pilot index is required.")

    if len(np.unique(pilot_array)) != len(pilot_array):
        raise ValueError("Pilot indices must be unique.")

    if np.any(pilot_array < 0) or np.any(pilot_array >= fft_size):
        raise ValueError("Pilot index is outside the resource grid.")

    pilot_value = complex(pilot_symbol)

    if abs(pilot_value) < 1e-12:
        raise ValueError("Pilot symbol cannot be zero.")

    pilot_set = set(pilot_array.tolist())

    available_data_indices = np.asarray(
        [
            index
            for index in range(fft_size)
            if index not in pilot_set
        ],
        dtype=np.int64,
    )

    if len(symbol_array) > len(available_data_indices):
        raise ValueError(
            f"Only {len(available_data_indices)} data subcarriers "
            "are available after reserving pilots."
        )

    used_data_indices = available_data_indices[: len(symbol_array)]

    grid = np.zeros(fft_size, dtype=np.complex128)
    grid[pilot_array] = pilot_value
    grid[used_data_indices] = symbol_array

    return grid, used_data_indices
