import numpy as np

from src.receiver.demapper import qpsk_demodulate
from src.receiver.ofdm_rx import ofdm_demodulate
from src.transmitter.modulator import qpsk_modulate
from src.transmitter.ofdm_tx import ofdm_modulate


def test_complete_qpsk_ofdm_chain_without_channel():
    original_bits = np.array(
        [0, 0, 0, 1, 1, 1, 1, 0],
        dtype=np.int8,
    )

    transmitted_symbols = qpsk_modulate(original_bits)

    transmitted_signal = ofdm_modulate(
        transmitted_symbols,
        fft_size=8,
        cp_length=2,
    )

    received_symbols = ofdm_demodulate(
        transmitted_signal,
        fft_size=8,
        cp_length=2,
        num_data_symbols=len(transmitted_symbols),
    )

    recovered_bits = qpsk_demodulate(received_symbols)

    np.testing.assert_array_equal(
        recovered_bits,
        original_bits,
    )
