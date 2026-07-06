"""Generate link-adaptation data with fading and interference."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from src.link_simulator import MODULATIONS, simulate_link_frame
from src.ml.modulation_selector import (
    select_best_modulation_from_ber,
)


FFT_SIZE = 64
NUM_SAMPLES = 600
PROBE_FRAMES = 4
LABEL_FRAMES = 12
TARGET_BER = 0.01


def generate_channel_coefficient(
    rng: np.random.Generator,
) -> complex:
    """Generate a stable Rayleigh fading coefficient."""
    while True:
        coefficient = complex(
            (
                rng.standard_normal()
                + 1j * rng.standard_normal()
            )
            / np.sqrt(2)
        )

        # Avoid numerically unstable extremely deep fades for now.
        if abs(coefficient) >= 0.15:
            return coefficient


def measure_probe(
    snr_db: float,
    channel_coefficient: complex,
    interference_to_signal_db: float | None,
    interference_frequency: float,
    interference_phase: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Measure receiver-observable features using QPSK probes."""
    ber_values: list[float] = []
    evm_values: list[float] = []
    pilot_error_values: list[float] = []
    channel_magnitude_values: list[float] = []

    packet_errors = 0

    for _ in range(PROBE_FRAMES):
        result = simulate_link_frame(
            modulation="QPSK",
            snr_db=snr_db,
            channel_coefficient=channel_coefficient,
            interference_to_signal_db=(
                interference_to_signal_db
            ),
            interference_frequency=interference_frequency,
            interference_phase_rad=interference_phase,
            rng=rng,
        )

        ber_values.append(result.ber)
        evm_values.append(result.evm_percent)
        pilot_error_values.append(
            result.pilot_error_percent
        )
        channel_magnitude_values.append(
            abs(result.estimated_channel)
        )

        if result.ber > 0:
            packet_errors += 1

    return {
        "probe_ber": float(np.mean(ber_values)),
        "probe_evm_percent": float(
            np.mean(evm_values)
        ),
        "packet_error_rate": (
            packet_errors / PROBE_FRAMES
        ),
        "probe_pilot_error_percent": float(
            np.mean(pilot_error_values)
        ),
        "estimated_channel_magnitude": float(
            np.mean(channel_magnitude_values)
        ),
    }


def measure_modulation_ber(
    modulation: str,
    snr_db: float,
    channel_coefficient: complex,
    interference_to_signal_db: float | None,
    interference_frequency: float,
    interference_phase: float,
    rng: np.random.Generator,
) -> float:
    """Measure average BER for one modulation."""
    ber_values: list[float] = []

    for _ in range(LABEL_FRAMES):
        result = simulate_link_frame(
            modulation=modulation,
            snr_db=snr_db,
            channel_coefficient=channel_coefficient,
            interference_to_signal_db=(
                interference_to_signal_db
            ),
            interference_frequency=interference_frequency,
            interference_phase_rad=interference_phase,
            rng=rng,
        )

        ber_values.append(result.ber)

    return float(np.mean(ber_values))


def main() -> None:
    """Generate and save the new training dataset."""
    rng = np.random.default_rng(2026)

    output_path = Path(
        "datasets/link_adaptation.csv"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()

    for sample_index in range(NUM_SAMPLES):
        snr_db = float(rng.uniform(0, 30))

        channel_coefficient = (
            generate_channel_coefficient(rng)
        )

        interference_present = bool(
            rng.random() < 0.60
        )

        if interference_present:
            interference_to_signal_db = float(
                rng.uniform(-30, -5)
            )

            interference_bin = int(
                rng.integers(
                    -FFT_SIZE // 2,
                    FFT_SIZE // 2,
                )
            )

            interference_frequency = (
                interference_bin / FFT_SIZE
            )

            interference_phase = float(
                rng.uniform(0, 2 * np.pi)
            )
        else:
            interference_to_signal_db = None
            interference_bin = -1
            interference_frequency = 0.0
            interference_phase = 0.0

        probe = measure_probe(
            snr_db=snr_db,
            channel_coefficient=channel_coefficient,
            interference_to_signal_db=(
                interference_to_signal_db
            ),
            interference_frequency=interference_frequency,
            interference_phase=interference_phase,
            rng=rng,
        )

        ber_by_modulation = {
            modulation: measure_modulation_ber(
                modulation=modulation,
                snr_db=snr_db,
                channel_coefficient=channel_coefficient,
                interference_to_signal_db=(
                    interference_to_signal_db
                ),
                interference_frequency=(
                    interference_frequency
                ),
                interference_phase=interference_phase,
                rng=rng,
            )
            for modulation in MODULATIONS
        }

        selected_modulation = (
            select_best_modulation_from_ber(
                ber_by_modulation,
                target_ber=TARGET_BER,
            )
        )

        selected_bits_per_symbol = MODULATIONS[
            selected_modulation
        ][0]

        rows.append(
            {
                "snr_db": snr_db,
                **probe,
                "qpsk_ber": ber_by_modulation["QPSK"],
                "qam16_ber": ber_by_modulation["16QAM"],
                "qam64_ber": ber_by_modulation["64QAM"],
                "selected_modulation": selected_modulation,
                "selected_bits_per_symbol": (
                    selected_bits_per_symbol
                ),
                # Ground-truth columns are stored only for analysis.
                "true_channel_magnitude": abs(
                    channel_coefficient
                ),
                "interference_present": int(
                    interference_present
                ),
                "interference_to_signal_db": (
                    interference_to_signal_db
                ),
                "interference_bin": interference_bin,
            }
        )

        label_counts[selected_modulation] += 1

        if (sample_index + 1) % 100 == 0:
            print(
                f"Generated {sample_index + 1}/"
                f"{NUM_SAMPLES} samples"
            )

    fieldnames = [
        "snr_db",
        "probe_evm_percent",
        "probe_ber",
        "packet_error_rate",
        "probe_pilot_error_percent",
        "estimated_channel_magnitude",
        "qpsk_ber",
        "qam16_ber",
        "qam64_ber",
        "selected_modulation",
        "selected_bits_per_symbol",
        "true_channel_magnitude",
        "interference_present",
        "interference_to_signal_db",
        "interference_bin",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDataset saved to {output_path}")
    print("Label distribution:")

    for modulation in MODULATIONS:
        print(
            f"  {modulation}: "
            f"{label_counts[modulation]}"
        )


if __name__ == "__main__":
    main()
