"""Bit-error-rate measurement."""

import numpy as np
from numpy.typing import ArrayLike


def calculate_ber(
    transmitted_bits: ArrayLike,
    received_bits: ArrayLike,
) -> float:
    """Calculate the fraction of incorrectly recovered bits."""
    transmitted = np.asarray(transmitted_bits, dtype=np.int8)
    received = np.asarray(received_bits, dtype=np.int8)

    if transmitted.ndim != 1 or received.ndim != 1:
        raise ValueError("Bit sequences must be one-dimensional.")

    if transmitted.size != received.size:
        raise ValueError("Bit sequences must have equal lengths.")

    if transmitted.size == 0:
        raise ValueError("Bit sequences cannot be empty.")

    if not np.all(np.isin(transmitted, [0, 1])):
        raise ValueError("Transmitted data must contain only 0 and 1.")

    if not np.all(np.isin(received, [0, 1])):
        raise ValueError("Received data must contain only 0 and 1.")

    return float(np.mean(transmitted != received))
