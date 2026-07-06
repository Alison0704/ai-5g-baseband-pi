import numpy as np

from src.receiver.demapper import qpsk_demodulate
from src.transmitter.modulator import qpsk_modulate

def test_qpsk_demodulation_mapping():
    symbols = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ]
    ) / np.sqrt(2)

    expected = np.array([0, 0, 0, 1, 1, 1, 1, 0])

    actual = qpsk_demodulate(symbols)

    np.testing.assert_array_equal(actual, expected)


def test_qpsk_modulation_demodulation_round_trip():
    original_bits = np.array(
        [0, 1, 1, 0, 0, 0, 1, 1],
        dtype=np.int8,
    )

    symbols = qpsk_modulate(original_bits)
    recovered_bits = qpsk_demodulate(symbols)

    np.testing.assert_array_equal(recovered_bits, original_bits)


def test_qpsk_demodulation_with_small_noise():
    original_bits = np.array([0, 0, 0, 1, 1, 1, 1, 0])

    symbols = qpsk_modulate(original_bits)

    noise = np.array(
        [
            0.05 + 0.02j,
            -0.03 + 0.04j,
            0.02 - 0.03j,
            -0.04 - 0.02j,
        ]
    )

    recovered_bits = qpsk_demodulate(symbols + noise)

    np.testing.assert_array_equal(recovered_bits, original_bits)


def test_empty_symbol_sequence():
    result = qpsk_demodulate([])

    assert result.size == 0
