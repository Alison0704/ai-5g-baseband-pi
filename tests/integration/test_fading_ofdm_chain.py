import numpy as np

from src.channel.awgn import add_awgn
from src.channel.fading import apply_flat_rayleigh_fading
from src.metrics.ber import calculate_ber
from src.receiver.demapper import qam16_demodulate
from src.receiver.equalizer import equalize_flat_channel
from src.receiver.ofdm_rx import ofdm_demodulate
from src.transmitter.modulator import qam16_modulate
from src.transmitter.ofdm_tx import ofdm_modulate


def test_qam16_ofdm_with_fading_and_equalization():
    rng = np.random.default_rng(123)

    fft_size = 64
    cp_length = 16

    transmitted_bits = rng.integers(
        0,
        2,
        size=fft_size * 4,
        dtype=np.int8,
    )

    transmitted_symbols = qam16_modulate(
        transmitted_bits
    )

    transmitted_signal = ofdm_modulate(
        transmitted_symbols,
        fft_size=fft_size,
        cp_length=cp_length,
    )

    faded_signal, channel_coefficient = (
        apply_flat_rayleigh_fading(
            transmitted_signal,
            coefficient=0.4 + 0.7j,
        )
    )

    received_signal = add_awgn(
        faded_signal,
        snr_db=60,
        rng=rng,
    )

    received_symbols = ofdm_demodulate(
        received_signal,
        fft_size=fft_size,
        cp_length=cp_length,
        num_data_symbols=fft_size,
    )

    equalized_symbols = equalize_flat_channel(
        received_symbols,
        channel_coefficient,
    )

    recovered_bits = qam16_demodulate(
        equalized_symbols
    )

    assert calculate_ber(
        transmitted_bits,
        recovered_bits,
    ) == 0.0
