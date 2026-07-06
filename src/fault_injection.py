"""Controlled fault injection for OFDM receiver diagnosis."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.channel.awgn import add_awgn
from src.channel.fading import apply_flat_rayleigh_fading
from src.link_simulator import (
    MODULATIONS,
    calculate_pilot_error_percent,
)
from src.metrics.ber import calculate_ber
from src.metrics.evm import calculate_evm_rms, evm_to_percent
from src.receiver.channel_estimator import estimate_flat_channel
from src.receiver.equalizer import equalize_flat_channel
from src.receiver.ofdm_rx import ofdm_demodulate
from src.transmitter.ofdm_tx import add_cyclic_prefix
from src.transmitter.resource_grid import map_data_and_pilots


FAULT_TYPES = (
    "NONE",
    "SYMBOL_GAIN_ERROR",
    "PILOT_CORRUPTION",
    "CHANNEL_ESTIMATION_BIAS",
    "EQUALIZER_BYPASS",
    "IQ_SWAP",
)


@dataclass
class FaultFrameResult:
    """Measurements collected from one fault-injected frame."""

    fault_type: str
    modulation: str
    ber: float
    evm_percent: float
    pilot_error_percent: float
    average_symbol_power: float
    estimated_channel_magnitude: float
    channel_estimation_error_percent: float


def simulate_faulty_link_frame(
    fault_type: str,
    modulation: str = "16QAM",
    snr_db: float = 30.0,
    channel_coefficient: complex = 0.8 + 0.3j,
    rng: np.random.Generator | None = None,
    fft_size: int = 64,
    cp_length: int = 16,
    pilot_indices: Sequence[int] = (7, 21, 43, 57),
    pilot_symbol: complex = 1 + 0j,
) -> FaultFrameResult:
    """Simulate one frame containing a selected processing fault."""
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"Unsupported fault type: {fault_type}")

    if modulation not in MODULATIONS:
        raise ValueError(f"Unsupported modulation: {modulation}")

    generator = rng or np.random.default_rng()

    bits_per_symbol, modulate, demodulate = MODULATIONS[
        modulation
    ]

    data_subcarriers = fft_size - len(set(pilot_indices))

    transmitted_bits = generator.integers(
        0,
        2,
        size=data_subcarriers * bits_per_symbol,
        dtype=np.int8,
    )

    ideal_symbols = modulate(transmitted_bits)
    transmitted_symbols = ideal_symbols.copy()

    if fault_type == "SYMBOL_GAIN_ERROR":
        transmitted_symbols *= 1.8

    resource_grid, data_indices = map_data_and_pilots(
        symbols=transmitted_symbols,
        fft_size=fft_size,
        pilot_indices=pilot_indices,
        pilot_symbol=pilot_symbol,
    )

    if fault_type == "PILOT_CORRUPTION":
        corrupted_index = int(pilot_indices[0])
        resource_grid[corrupted_index] *= -1

    time_signal = np.fft.ifft(resource_grid)

    transmitted_signal = add_cyclic_prefix(
        time_signal,
        cp_length,
    )

    faded_signal, true_channel = apply_flat_rayleigh_fading(
        transmitted_signal,
        coefficient=channel_coefficient,
    )

    received_signal = add_awgn(
        faded_signal,
        snr_db=snr_db,
        rng=generator,
    )

    received_grid = ofdm_demodulate(
        received_signal,
        fft_size=fft_size,
        cp_length=cp_length,
    )

    estimated_channel = estimate_flat_channel(
        received_grid=received_grid,
        pilot_indices=pilot_indices,
        pilot_symbol=pilot_symbol,
    )

    if fault_type == "CHANNEL_ESTIMATION_BIAS":
        estimated_channel *= 1.5 + 0.3j

    received_data_symbols = received_grid[data_indices]

    if fault_type == "EQUALIZER_BYPASS":
        equalized_symbols = received_data_symbols
    else:
        equalized_symbols = equalize_flat_channel(
            received_data_symbols,
            estimated_channel,
        )

    if fault_type == "IQ_SWAP":
        equalized_symbols = (
            equalized_symbols.imag
            + 1j * equalized_symbols.real
        )

    received_bits = demodulate(equalized_symbols)

    ber = calculate_ber(
        transmitted_bits,
        received_bits,
    )

    evm_percent = evm_to_percent(
        calculate_evm_rms(
            ideal_symbols,
            equalized_symbols,
        )
    )

    pilot_error_percent = calculate_pilot_error_percent(
        received_grid=received_grid,
        pilot_indices=pilot_indices,
        estimated_channel=estimated_channel,
        pilot_symbol=pilot_symbol,
    )

    average_symbol_power = float(
        np.mean(np.abs(equalized_symbols) ** 2)
    )

    channel_estimation_error_percent = float(
        100
        * abs(estimated_channel - true_channel)
        / max(abs(true_channel), 1e-12)
    )

    return FaultFrameResult(
        fault_type=fault_type,
        modulation=modulation,
        ber=ber,
        evm_percent=evm_percent,
        pilot_error_percent=pilot_error_percent,
        average_symbol_power=average_symbol_power,
        estimated_channel_magnitude=abs(estimated_channel),
        channel_estimation_error_percent=(
            channel_estimation_error_percent
        ),
    )
