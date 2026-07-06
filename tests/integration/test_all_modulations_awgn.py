import numpy as np
import pytest

from src.channel.awgn import add_awgn
from src.metrics.ber import calculate_ber
from src.receiver.demapper import (
    qam16_demodulate,
    qam64_demodulate,
    qpsk_demodulate,
)
from src.receiver.ofdm_rx import ofdm_demodulate
from src.transmitter.modulator import (
    qam16_modulate,
    qam64_modulate,
    qpsk_modulate,
)
from src.transmitter.ofdm_tx import ofdm_modulate


@pytest.mark.parametrize(
    (
        "bits_per_symbol",
        "modulate",
        "demodulate",
    ),
    [
        (2, qpsk_modulate, qpsk_demodulate),
        (4, qam16_modulate, qam16_demodulate),
        (6, qam64_modulate, qam64_demodulate),
    ],
)
def test_modulation_chain_at_high_snr(
    bits_per_symbol,
    modulate,
    demodulate,
):
    rng = np.random.default_rng(123)
    fft_size = 64
    cp_length = 16

    transmitted_bits = rng.integers(
        0,
        2,
        size=fft_size * bits_per_symbol,
        dtype=np.int8,
    )

    transmitted_symbols = modulate(transmitted_bits)

    transmitted_signal = ofdm_modulate(
        transmitted_symbols,
        fft_size=fft_size,
        cp_length=cp_length,
    )

    received_signal = add_awgn(
        transmitted_signal,
        snr_db=60,
        rng=rng,
    )

    received_symbols = ofdm_demodulate(
        received_signal,
        fft_size=fft_size,
        cp_length=cp_length,
        num_data_symbols=fft_size,
    )

    received_bits = demodulate(received_symbols)

    assert calculate_ber(
        transmitted_bits,
        received_bits,
    ) == 0.0
