"""Pilot-based wireless-channel estimation."""

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike


def estimate_flat_channel(
    received_grid: ArrayLike,
    pilot_indices: Sequence[int],
    pilot_symbol: complex = 1 + 0j,
) -> complex:
    """
    Estimate one flat channel coefficient using known pilots.

    For every pilot:
        estimate = received pilot / transmitted pilot

    The final result is the average of all pilot estimates.
    """
    grid = np.asarray(received_grid, dtype=np.complex128)
    indices = np.asarray(pilot_indices, dtype=np.int64)

    if grid.ndim != 1:
        raise ValueError("Received grid must be one-dimensional.")

    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("At least one pilot index is required.")

    if len(np.unique(indices)) != len(indices):
        raise ValueError("Pilot indices must be unique.")

    if np.any(indices < 0) or np.any(indices >= len(grid)):
        raise ValueError("Pilot index is outside the received grid.")

    known_pilot = complex(pilot_symbol)

    if abs(known_pilot) < 1e-12:
        raise ValueError("Pilot symbol cannot be zero.")

    received_pilots = grid[indices]
    individual_estimates = received_pilots / known_pilot

    return complex(np.mean(individual_estimates))
