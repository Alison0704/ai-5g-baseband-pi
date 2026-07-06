import numpy as np

from src.channel.awgn import add_awgn
from src.channel.fading import apply_flat_rayleigh_fading
from src.channel.interference import add_tone_interference
from src.metrics.ber import calculate_ber
from src.receiver.channel_estimator import estimate_flat_channel
from src.receiver.demapper import qam16_demodulate
from src.receiver.equalizer import equalize_flat_channel
from src.receiver.ofdm_rx import ofdm_demodulate
from src.transmitter.modulator import qam16_modulate
from src.transmitter.ofdm_tx import ofdm_modulate_with_pilots


def test_ofdm_chain_with_weak_interference():
    rng = np.random.default_rng(123)

    fft_size = 64
    cp_length = 16
    pilot_indices = (7, 21, 43, 57)

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
        coefficient=0.8 + 0.3j,
    )

    interfered_signal, _ = add_tone_interference(
        faded_signal,
        interference_to_signal_db=-35,
        normalized_frequency=10 / fft_size,
        phase_rad=0.25,
    )

    received_signal = add_awgn(
        interfered_signal,
        snr_db=40,
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

    equalized_symbols = equalize_flat_channel(
        received_grid[data_indices],
        estimated_channel,
    )

    received_bits = qam16_demodulate(
        equalized_symbols
    )

    ber = calculate_ber(
        transmitted_bits,
        received_bits,
    )

    assert ber == 0.0
