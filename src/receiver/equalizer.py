"""Wireless-channel equalization functions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def equalize_flat_channel(
    received_symbols: ArrayLike,
    channel_coefficient: complex,
) -> NDArray[np.complex128]:
    """
    Remove a known flat channel coefficient from received symbols.

    This temporarily assumes perfect channel knowledge. A pilot-based
    channel estimator will replace that assumption in the next step.
    """
    symbol_array = np.asarray(
        received_symbols,
        dtype=np.complex128,
    )

    if symbol_array.ndim != 1:
        raise ValueError("Symbols must be one-dimensional.")

    coefficient = complex(channel_coefficient)

    if not (
        np.isfinite(coefficient.real)
        and np.isfinite(coefficient.imag)
    ):
        raise ValueError("Channel coefficient must be finite.")

    if abs(coefficient) < 1e-12:
        raise ValueError(
            "Channel coefficient is too small for equalization."
        )

    return symbol_array / coefficient
