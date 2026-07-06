"""Digital symbol demapping functions for the OFDM receiver."""

from typing import Iterable

import numpy as np
from numpy.typing import NDArray


def _validate_symbols(
    symbols: Iterable[complex],
) -> NDArray[np.complex128]:
    """Convert and validate a symbol sequence."""
    symbol_array = np.asarray(list(symbols), dtype=np.complex128)

    if symbol_array.ndim != 1:
        raise ValueError("Symbols must be a one-dimensional sequence.")

    return symbol_array


def qpsk_demodulate(
    symbols: Iterable[complex],
) -> NDArray[np.int8]:
    """Convert normalized QPSK symbols into bits."""
    symbol_array = _validate_symbols(symbols)

    if len(symbol_array) == 0:
        return np.array([], dtype=np.int8)

    bits = np.empty(len(symbol_array) * 2, dtype=np.int8)

    bits[0::2] = (symbol_array.imag < 0).astype(np.int8)
    bits[1::2] = (symbol_array.real < 0).astype(np.int8)

    return bits


def _qam16_level_to_bits(value: float) -> tuple[int, int]:
    """Convert one 16-QAM axis value into bits."""
    if value >= 2:
        return 0, 0

    if value >= 0:
        return 0, 1

    if value >= -2:
        return 1, 1

    return 1, 0


def qam16_demodulate(
    symbols: Iterable[complex],
) -> NDArray[np.int8]:
    """Convert normalized 16-QAM symbols into bits."""
    symbol_array = _validate_symbols(symbols)

    if len(symbol_array) == 0:
        return np.array([], dtype=np.int8)

    scaled_symbols = symbol_array * np.sqrt(10)
    recovered_bits: list[int] = []

    for symbol in scaled_symbols:
        recovered_bits.extend(_qam16_level_to_bits(symbol.real))
        recovered_bits.extend(_qam16_level_to_bits(symbol.imag))

    return np.asarray(recovered_bits, dtype=np.int8)


def _qam64_level_to_bits(value: float) -> tuple[int, int, int]:
    """Convert one 64-QAM axis value into bits."""
    if value >= 6:
        return 0, 0, 0

    if value >= 4:
        return 0, 0, 1

    if value >= 2:
        return 0, 1, 1

    if value >= 0:
        return 0, 1, 0

    if value >= -2:
        return 1, 1, 0

    if value >= -4:
        return 1, 1, 1

    if value >= -6:
        return 1, 0, 1

    return 1, 0, 0


def qam64_demodulate(
    symbols: Iterable[complex],
) -> NDArray[np.int8]:
    """Convert normalized 64-QAM symbols into bits."""
    symbol_array = _validate_symbols(symbols)

    if len(symbol_array) == 0:
        return np.array([], dtype=np.int8)

    scaled_symbols = symbol_array * np.sqrt(42)
    recovered_bits: list[int] = []

    for symbol in scaled_symbols:
        recovered_bits.extend(_qam64_level_to_bits(symbol.real))
        recovered_bits.extend(_qam64_level_to_bits(symbol.imag))

    return np.asarray(recovered_bits, dtype=np.int8)
