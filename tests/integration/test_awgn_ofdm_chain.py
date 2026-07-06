import numpy as np

from src.channel.awgn import add_awgn
from src.metrics.ber import calculate_ber
from src.receiver.demapper import qpsk_demodulate
from src.receiver.ofdm_rx import ofdm_demodulate
from src.transmitter.modulator import qpsk_modulate
from src.transmitter.ofdm_tx import ofdm_modulate


def test_qpsk_ofdm_chain_at_high_snr():
    rng = np.random.default_rng(123)

    original_bits = rng.integers(
        0,
        2,
        size=256,
        dtype=np.int8,
    )

    transmitted_symbols = qpsk_modulate(original_bits)

    transmitted_signal = ofdm_modulate(
        transmitted_symbols,
        fft_size=128,
        cp_length=16,
    )

    received_signal = add_awgn(
        transmitted_signal,
        snr_db=60,
        rng=rng,
    )

    received_symbols = ofdm_demodulate(
        received_signal,
        fft_size=128,
        cp_length=16,
        num_data_symbols=len(transmitted_symbols),
    )

    recovered_bits = qpsk_demodulate(received_symbols)

    ber = calculate_ber(original_bits, recovered_bits)

    assert ber == 0.0
