import numpy as np
import pytest

from src.transmitter.resource_grid import map_data_to_subcarriers


def test_symbols_are_mapped_to_grid():
    symbols = np.array([1 + 1j, -1 + 1j])

    grid = map_data_to_subcarriers(symbols, fft_size=8)

    np.testing.assert_array_equal(grid[:2], symbols)
    np.testing.assert_array_equal(grid[2:], np.zeros(6))


def test_rejects_too_many_symbols():
    symbols = np.ones(9, dtype=np.complex128)

    with pytest.raises(ValueError):
        map_data_to_subcarriers(symbols, fft_size=8)


from src.transmitter.resource_grid import map_data_and_pilots


def test_data_and_pilot_mapping():
    data_symbols = np.array(
        [1 + 1j, -1 + 1j],
        dtype=np.complex128,
    )

    grid, data_indices = map_data_and_pilots(
        symbols=data_symbols,
        fft_size=8,
        pilot_indices=(1, 5),
        pilot_symbol=1 + 0j,
    )

    np.testing.assert_array_equal(
        grid[[1, 5]],
        np.array([1 + 0j, 1 + 0j]),
    )

    np.testing.assert_array_equal(
        data_indices,
        np.array([0, 2]),
    )

    np.testing.assert_array_equal(
        grid[data_indices],
        data_symbols,
    )
