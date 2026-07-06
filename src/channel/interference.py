"""Interference models for simulated wireless channels."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def add_tone_interference(
    signal: ArrayLike,
    interference_to_signal_db: float,
    normalized_frequency: float = 0.125,
    phase_rad: float = 0.0,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """
    Add complex sinusoidal interference to a signal.

    Args:
        signal:
            Complex time-domain waveform.
        interference_to_signal_db:
            Interference-to-signal power ratio in decibels.

            -20 dB means the interference power is 100 times weaker
            than the signal power.

            0 dB means equal signal and interference power.
        normalized_frequency:
            Tone frequency in cycles per sample. Valid range is
            -0.5 <= frequency < 0.5.
        phase_rad:
            Initial tone phase in radians.

    Returns:
        Signal containing interference.
        Generated interference samples.
    """
    signal_array = np.asarray(
        signal,
        dtype=np.complex128,
    )

    if signal_array.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal_array.size == 0:
        empty = signal_array.copy()
        return empty, empty

    if not np.isfinite(interference_to_signal_db):
        raise ValueError(
            "Interference-to-signal ratio must be finite."
        )

    if not -0.5 <= normalized_frequency < 0.5:
        raise ValueError(
            "Normalized frequency must be in [-0.5, 0.5)."
        )

    if not np.isfinite(phase_rad):
        raise ValueError("Phase must be finite.")

    signal_power = float(
        np.mean(np.abs(signal_array) ** 2)
    )

    if signal_power <= 0:
        raise ValueError(
            "Signal power must be greater than zero."
        )

    power_ratio = 10 ** (
        interference_to_signal_db / 10
    )

    interference_power = signal_power * power_ratio
    amplitude = np.sqrt(interference_power)

    sample_indices = np.arange(signal_array.size)

    interference = amplitude * np.exp(
        1j
        * (
            2
            * np.pi
            * normalized_frequency
            * sample_indices
            + phase_rad
        )
    )

    return signal_array + interference, interference
