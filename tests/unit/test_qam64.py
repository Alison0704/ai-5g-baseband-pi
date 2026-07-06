from itertools import product

import numpy as np
import pytest

from src.receiver.demapper import qam64_demodulate
from src.transmitter.modulator import qam64_modulate


def test_qam64_known_mapping():
    bits = [
        0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 1, 0,
        1, 0, 0, 1, 0, 0,
    ]

    expected = np.array(
        [
            7 + 7j,
            1 - 1j,
            -7 - 7j,
        ],
        dtype=np.complex128,
    ) / np.sqrt(42)

    actual = qam64_modulate(bits)

    np.testing.assert_allclose(actual, expected)


def test_qam64_average_power():
    all_groups = list(product([0, 1], repeat=6))
    bits = np.asarray(all_groups, dtype=np.int8).reshape(-1)

    symbols = qam64_modulate(bits)

    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0)


def test_qam64_round_trip():
    rng = np.random.default_rng(123)

    original_bits = rng.integers(
        0,
        2,
        size=600,
        dtype=np.int8,
    )

    symbols = qam64_modulate(original_bits)
    recovered_bits = qam64_demodulate(symbols)

    np.testing.assert_array_equal(
        recovered_bits,
        original_bits,
    )


def test_qam64_rejects_incomplete_symbol():
    with pytest.raises(ValueError):
        qam64_modulate([0, 1, 0, 1])
