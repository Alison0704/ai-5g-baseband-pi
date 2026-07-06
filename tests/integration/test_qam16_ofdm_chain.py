import numpy as np

from src.receiver.demapper import qam16_demodulate
from src.receiver.ofdm_rx import ofdm_demodulate
from src.transmitter.modulator import qam16_modulate
from src.transmitter.ofdm_tx import ofdm_modulate


def test_complete_qam16_ofdm_chain_without_channel():
    rng = np.random.default_rng(123)

    original_bits = rng.integers(
        0,
        2,
        size=256,
        dtype=np.int8,
    )

    transmitted_symbols = qam16_modulate(original_bits)

    transmitted_signal = ofdm_modulate(
        transmitted_symbols,
        fft_size=64,
        cp_length=16,
    )

    received_symbols = ofdm_demodulate(
        transmitted_signal,
        fft_size=64,
        cp_length=16,
        num_data_symbols=len(transmitted_symbols),
    )

    recovered_bits = qam16_demodulate(received_symbols)

    np.testing.assert_array_equal(
        recovered_bits,
        original_bits,
    )
