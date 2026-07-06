import numpy as np
import pytest

from src.receiver.ofdm_rx import (
    ofdm_demodulate,
    remove_cyclic_prefix,
)
from src.transmitter.modulator import qpsk_modulate
from src.transmitter.ofdm_tx import ofdm_modulate


def test_remove_cyclic_prefix():
    received = np.array([3, 4, 1, 2, 3, 4])

    result = remove_cyclic_prefix(
        received,
        fft_size=4,
        cp_length=2,
    )

    expected = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(result, expected)


def test_rejects_incorrect_ofdm_symbol_length():
    with pytest.raises(ValueError):
        remove_cyclic_prefix(
            [1, 2, 3],
            fft_size=4,
            cp_length=2,
        )


def test_ofdm_transmitter_receiver_round_trip():
    symbols = qpsk_modulate(
        [0, 0, 0, 1, 1, 1, 1, 0]
    )

    transmitted = ofdm_modulate(
        symbols,
        fft_size=8,
        cp_length=2,
    )

    recovered = ofdm_demodulate(
        transmitted,
        fft_size=8,
        cp_length=2,
        num_data_symbols=len(symbols),
    )

    np.testing.assert_allclose(
        recovered,
        symbols,
        atol=1e-12,
    )
