"""End-to-end OFDM wireless-link simulation."""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from src.channel.awgn import add_awgn
from src.channel.fading import apply_flat_rayleigh_fading
from src.channel.interference import add_tone_interference
from src.metrics.ber import calculate_ber
from src.metrics.evm import calculate_evm_rms, evm_to_percent
from src.receiver.channel_estimator import estimate_flat_channel
from src.receiver.demapper import (
    qam16_demodulate,
    qam64_demodulate,
    qpsk_demodulate,
)
from src.receiver.equalizer import equalize_flat_channel
from src.receiver.ofdm_rx import ofdm_demodulate
from src.transmitter.modulator import (
    qam16_modulate,
    qam64_modulate,
    qpsk_modulate,
)
from src.transmitter.ofdm_tx import ofdm_modulate_with_pilots


Modulator = Callable[
    [NDArray[np.int8]],
    NDArray[np.complex128],
]

Demodulator = Callable[
    [NDArray[np.complex128]],
    NDArray[np.int8],
]


MODULATIONS: dict[
    str,
    tuple[int, Modulator, Demodulator],
] = {
    "QPSK": (
        2,
        qpsk_modulate,
        qpsk_demodulate,
    ),
    "16QAM": (
        4,
        qam16_modulate,
        qam16_demodulate,
    ),
    "64QAM": (
        6,
        qam64_modulate,
        qam64_demodulate,
    ),
}


@dataclass
class LinkFrameResult:
    """Measurements produced by one simulated wireless frame."""

    modulation: str
    ber: float
    evm_percent: float
    pilot_error_percent: float
    estimated_channel: complex
    true_channel: complex
    transmitted_bits: NDArray[np.int8]
    received_bits: NDArray[np.int8]


def calculate_pilot_error_percent(
    received_grid: NDArray[np.complex128],
    pilot_indices: Sequence[int],
    estimated_channel: complex,
    pilot_symbol: complex,
) -> float:
    """Calculate residual error across the pilot subcarriers."""
    indices = np.asarray(pilot_indices, dtype=np.int64)
    received_pilots = received_grid[indices]

    predicted_pilots = np.full(
        len(indices),
        estimated_channel * pilot_symbol,
        dtype=np.complex128,
    )

    reference_power = float(
        np.mean(np.abs(received_pilots) ** 2)
    )

    if reference_power <= 0:
        raise ValueError(
            "Received pilot power must be greater than zero."
        )

    error_power = float(
        np.mean(
            np.abs(
                received_pilots - predicted_pilots
            ) ** 2
        )
    )

    return float(
        100 * np.sqrt(error_power / reference_power)
    )


def simulate_link_frame(
    modulation: str,
    snr_db: float,
    rng: np.random.Generator | None = None,
    fft_size: int = 64,
    cp_length: int = 16,
    pilot_indices: Sequence[int] = (7, 21, 43, 57),
    pilot_symbol: complex = 1 + 0j,
    channel_coefficient: complex | None = None,
    interference_to_signal_db: float | None = None,
    interference_frequency: float = 10 / 64,
    interference_phase_rad: float = 0.0,
) -> LinkFrameResult:
    """Simulate one complete OFDM transmission and reception."""
    if modulation not in MODULATIONS:
        raise ValueError(
            f"Unsupported modulation: {modulation}"
        )

    generator = (
        rng
        if rng is not None
        else np.random.default_rng()
    )

    bits_per_symbol, modulate, demodulate = (
        MODULATIONS[modulation]
    )

    number_of_data_subcarriers = (
        fft_size - len(set(pilot_indices))
    )

    if number_of_data_subcarriers <= 0:
        raise ValueError(
            "Pilot configuration leaves no data subcarriers."
        )

    transmitted_bits = generator.integers(
        0,
        2,
        size=number_of_data_subcarriers * bits_per_symbol,
        dtype=np.int8,
    )

    transmitted_symbols = modulate(transmitted_bits)

    transmitted_signal, data_indices = (
        ofdm_modulate_with_pilots(
            symbols=transmitted_symbols,
            fft_size=fft_size,
            cp_length=cp_length,
            pilot_indices=pilot_indices,
            pilot_symbol=pilot_symbol,
        )
    )

    faded_signal, true_channel = (
        apply_flat_rayleigh_fading(
            transmitted_signal,
            coefficient=channel_coefficient,
            rng=generator,
        )
    )

    # Noise power is measured relative to the faded signal.
    received_signal = add_awgn(
        faded_signal,
        snr_db=snr_db,
        rng=generator,
    )

    if interference_to_signal_db is not None:
        _, interference = add_tone_interference(
            faded_signal,
            interference_to_signal_db=(
                interference_to_signal_db
            ),
            normalized_frequency=interference_frequency,
            phase_rad=interference_phase_rad,
        )

        received_signal = received_signal + interference

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

    equalized_symbols = equalize_flat_channel(
        received_grid[data_indices],
        estimated_channel,
    )

    received_bits = demodulate(equalized_symbols)

    ber = calculate_ber(
        transmitted_bits,
        received_bits,
    )

    evm_percent = evm_to_percent(
        calculate_evm_rms(
            transmitted_symbols,
            equalized_symbols,
        )
    )

    pilot_error_percent = calculate_pilot_error_percent(
        received_grid=received_grid,
        pilot_indices=pilot_indices,
        estimated_channel=estimated_channel,
        pilot_symbol=pilot_symbol,
    )

    return LinkFrameResult(
        modulation=modulation,
        ber=ber,
        evm_percent=evm_percent,
        pilot_error_percent=pilot_error_percent,
        estimated_channel=estimated_channel,
        true_channel=true_channel,
        transmitted_bits=transmitted_bits,
        received_bits=received_bits,
    )
