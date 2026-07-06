import numpy as np

from src.transmitter.ofdm_tx import add_cyclic_prefix, ofdm_modulate
from src.transmitter.modulator import qpsk_modulate


def test_cyclic_prefix_matches_end_of_symbol():
    signal = np.array([1, 2, 3, 4], dtype=np.complex128)

    result = add_cyclic_prefix(signal, cp_length=2)

    expected = np.array([3, 4, 1, 2, 3, 4])
    np.testing.assert_array_equal(result, expected)


def test_ofdm_symbol_length():
    symbols = qpsk_modulate([0, 0, 0, 1])

    result = ofdm_modulate(
        symbols,
        fft_size=8,
        cp_length=2,
    )

    assert len(result) == 10


def test_ifft_fft_round_trip_without_channel():
    symbols = qpsk_modulate(
        [0, 0, 0, 1, 1, 1, 1, 0]
    )

    transmitted = ofdm_modulate(
        symbols,
        fft_size=8,
        cp_length=2,
    )

    time_signal = transmitted[2:]
    recovered_grid = np.fft.fft(time_signal)

    np.testing.assert_allclose(
        recovered_grid[: len(symbols)],
        symbols,
        atol=1e-12,
    )
