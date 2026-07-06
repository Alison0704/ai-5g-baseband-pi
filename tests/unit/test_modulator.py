import numpy as np
import pytest

from src.transmitter.modulator import qpsk_modulate

def test_qpsk_modulation_mapping():
    bits = [0, 0, 0, 1, 1, 1, 1, 0]

    expected = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ]
    ) / np.sqrt(2)

    actual = qpsk_modulate(bits)

    np.testing.assert_allclose(actual, expected)


def test_qpsk_average_power():
    bits = [0, 0, 0, 1, 1, 1, 1, 0]

    symbols = qpsk_modulate(bits)

    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0)


def test_qpsk_rejects_odd_number_of_bits():
    with pytest.raises(ValueError):
        qpsk_modulate([0, 1, 0])


def test_qpsk_rejects_invalid_bits():
    with pytest.raises(ValueError):
        qpsk_modulate([0, 2])
