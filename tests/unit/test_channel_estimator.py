import numpy as np
import pytest

from src.receiver.channel_estimator import estimate_flat_channel


def test_estimates_known_flat_channel():
    channel = 0.4 + 0.7j
    pilot_indices = (1, 4, 7)

    received_grid = np.zeros(8, dtype=np.complex128)
    received_grid[list(pilot_indices)] = channel

    estimate = estimate_flat_channel(
        received_grid=received_grid,
        pilot_indices=pilot_indices,
        pilot_symbol=1 + 0j,
    )

    assert estimate == pytest.approx(channel)


def test_estimator_averages_pilot_measurements():
    received_grid = np.zeros(4, dtype=np.complex128)
    received_grid[[0, 2]] = [
        0.9 + 0.1j,
        1.1 - 0.1j,
    ]

    estimate = estimate_flat_channel(
        received_grid,
        pilot_indices=(0, 2),
    )

    assert estimate == pytest.approx(1 + 0j)


def test_estimator_rejects_invalid_index():
    with pytest.raises(ValueError):
        estimate_flat_channel(
            received_grid=np.zeros(4),
            pilot_indices=(5,),
        )
