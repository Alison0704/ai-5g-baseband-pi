"""Error-vector-magnitude measurements."""

import numpy as np
from numpy.typing import ArrayLike


def calculate_evm_rms(
    reference_symbols: ArrayLike,
    received_symbols: ArrayLike,
) -> float:
    """
    Calculate RMS EVM as a decimal ratio.

    Example:
        0.10 means 10% EVM.
    """
    reference = np.asarray(reference_symbols, dtype=np.complex128)
    received = np.asarray(received_symbols, dtype=np.complex128)

    if reference.ndim != 1 or received.ndim != 1:
        raise ValueError("Symbol sequences must be one-dimensional.")

    if reference.size != received.size:
        raise ValueError("Symbol sequences must have equal lengths.")

    if reference.size == 0:
        raise ValueError("Symbol sequences cannot be empty.")

    reference_power = np.mean(np.abs(reference) ** 2)

    if reference_power <= 0:
        raise ValueError("Reference-symbol power must be greater than zero.")

    error_power = np.mean(np.abs(received - reference) ** 2)

    return float(np.sqrt(error_power / reference_power))


def evm_to_percent(evm_rms: float) -> float:
    """Convert decimal EVM into percentage EVM."""
    if evm_rms < 0:
        raise ValueError("EVM cannot be negative.")

    return evm_rms * 100.0
