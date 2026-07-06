from itertools import product

import numpy as np
import pytest

from src.receiver.demapper import qam16_demodulate
from src.transmitter.modulator import qam16_modulate


def test_qam16_known_mapping():
    bits = [
        0, 0, 0, 0,
        0, 1, 0, 1,
        1, 1, 1, 1,
        1, 0, 1, 0,
    ]

    expected = np.array(
        [
            3 + 3j,
            1 + 1j,
            -1 - 1j,
            -3 - 3j,
        ],
        dtype=np.complex128,
    ) / np.sqrt(10)

    actual = qam16_modulate(bits)

    np.testing.assert_allclose(actual, expected)


def test_qam16_average_power():
    all_groups = list(product([0, 1], repeat=4))
    bits = np.asarray(all_groups, dtype=np.int8).reshape(-1)

    symbols = qam16_modulate(bits)

    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0)


def test_qam16_round_trip():
    rng = np.random.default_rng(123)
    original_bits = rng.integers(
        0,
        2,
        size=400,
        dtype=np.int8,
    )

    symbols = qam16_modulate(original_bits)
    recovered_bits = qam16_demodulate(symbols)

    np.testing.assert_array_equal(
        recovered_bits,
        original_bits,
    )


def test_qam16_with_small_noise():
    original_bits = np.array(
        [0, 0, 0, 0, 0, 1, 1, 1],
        dtype=np.int8,
    )

    symbols = qam16_modulate(original_bits)

    noise = np.array(
        [0.02 - 0.01j, -0.03 + 0.02j],
        dtype=np.complex128,
    )

    recovered_bits = qam16_demodulate(symbols + noise)

    np.testing.assert_array_equal(
        recovered_bits,
        original_bits,
    )


def test_qam16_rejects_incomplete_symbol():
    with pytest.raises(ValueError):
        qam16_modulate([0, 1, 1])
