"""Digital modulation functions for the OFDM transmitter."""

from typing import Iterable

import numpy as np
from numpy.typing import NDArray


def _validate_bits(
    bits: Iterable[int],
    bits_per_symbol: int,
    modulation_name: str,
) -> NDArray[np.int8]:
    """Validate and convert an input bit sequence."""
    bit_array = np.asarray(list(bits), dtype=np.int8)

    if bit_array.ndim != 1:
        raise ValueError("Bits must be a one-dimensional sequence.")

    if not np.all(np.isin(bit_array, [0, 1])):
        raise ValueError("Bits must contain only 0 and 1.")

    if len(bit_array) % bits_per_symbol != 0:
        raise ValueError(
            f"{modulation_name} requires a multiple of "
            f"{bits_per_symbol} bits."
        )

    return bit_array


def qpsk_modulate(
    bits: Iterable[int],
) -> NDArray[np.complex128]:
    """Map bits to normalized Gray-coded QPSK symbols."""
    bit_array = _validate_bits(bits, 2, "QPSK")

    if len(bit_array) == 0:
        return np.array([], dtype=np.complex128)

    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }

    pairs = bit_array.reshape(-1, 2)

    symbols = np.array(
        [mapping[tuple(pair)] for pair in pairs],
        dtype=np.complex128,
    )

    return symbols / np.sqrt(2)


def qam16_modulate(
    bits: Iterable[int],
) -> NDArray[np.complex128]:
    """Map bits to normalized Gray-coded 16-QAM symbols."""
    bit_array = _validate_bits(bits, 4, "16-QAM")

    if len(bit_array) == 0:
        return np.array([], dtype=np.complex128)

    level_mapping = {
        (0, 0): 3,
        (0, 1): 1,
        (1, 1): -1,
        (1, 0): -3,
    }

    groups = bit_array.reshape(-1, 4)

    symbols = np.array(
        [
            level_mapping[(group[0], group[1])]
            + 1j * level_mapping[(group[2], group[3])]
            for group in groups
        ],
        dtype=np.complex128,
    )

    return symbols / np.sqrt(10)


def qam64_modulate(
    bits: Iterable[int],
) -> NDArray[np.complex128]:
    """Map bits to normalized Gray-coded 64-QAM symbols."""
    bit_array = _validate_bits(bits, 6, "64-QAM")

    if len(bit_array) == 0:
        return np.array([], dtype=np.complex128)

    level_mapping = {
        (0, 0, 0): 7,
        (0, 0, 1): 5,
        (0, 1, 1): 3,
        (0, 1, 0): 1,
        (1, 1, 0): -1,
        (1, 1, 1): -3,
        (1, 0, 1): -5,
        (1, 0, 0): -7,
    }

    groups = bit_array.reshape(-1, 6)

    symbols = np.array(
        [
            level_mapping[tuple(group[:3])]
            + 1j * level_mapping[tuple(group[3:])]
            for group in groups
        ],
        dtype=np.complex128,
    )

    # Average unnormalized 64-QAM symbol power is 42.
    return symbols / np.sqrt(42)
