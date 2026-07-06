import numpy as np

from src.channel.awgn import add_awgn
from src.channel.fading import apply_flat_rayleigh_fading
from src.metrics.ber import calculate_ber
from src.receiver.channel_estimator import estimate_flat_channel
from src.receiver.demapper import qam16_demodulate
from src.receiver.equalizer import equalize_flat_channel
from src.receiver.ofdm_rx import ofdm_demodulate
from src.transmitter.modulator import qam16_modulate
from src.transmitter.ofdm_tx import ofdm_modulate_with_pilots


def test_pilot_estimation_and_equalization():
    rng = np.random.default_rng(123)

    fft_size = 64
    cp_length = 16
    pilot_indices = (7, 21, 43, 57)
    channel_coefficient = 0.4 + 0.7j

    # Four pilots leave 60 data subcarriers.
    transmitted_bits = rng.integers(
        0,
        2,
        size=60 * 4,
        dtype=np.int8,
    )

    transmitted_symbols = qam16_modulate(
        transmitted_bits
    )

    transmitted_signal, data_indices = (
        ofdm_modulate_with_pilots(
            symbols=transmitted_symbols,
            fft_size=fft_size,
            cp_length=cp_length,
            pilot_indices=pilot_indices,
        )
    )

    faded_signal, _ = apply_flat_rayleigh_fading(
        transmitted_signal,
        coefficient=channel_coefficient,
    )

    received_signal = add_awgn(
        faded_signal,
        snr_db=60,
        rng=rng,
    )

    received_grid = ofdm_demodulate(
        received_signal,
        fft_size=fft_size,
        cp_length=cp_length,
    )

    estimated_channel = estimate_flat_channel(
        received_grid=received_grid,
        pilot_indices=pilot_indices,
    )

    received_data_symbols = received_grid[data_indices]

    equalized_symbols = equalize_flat_channel(
        received_data_symbols,
        estimated_channel,
    )

    recovered_bits = qam16_demodulate(
        equalized_symbols
    )

    assert abs(
        estimated_channel - channel_coefficient
    ) < 0.01

    assert calculate_ber(
        transmitted_bits,
        recovered_bits,
    ) == 0.0
