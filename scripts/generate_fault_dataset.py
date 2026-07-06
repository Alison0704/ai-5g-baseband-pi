"""Generate labeled data for OFDM fault diagnosis."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from src.fault_injection import (
    FAULT_TYPES,
    simulate_faulty_link_frame,
)
from src.link_simulator import MODULATIONS


SAMPLES_PER_FAULT = 150
OUTPUT_PATH = Path("datasets/fault_diagnosis.csv")
BASE_SEED = 2026


def generate_channel_coefficient(
    rng: np.random.Generator,
) -> complex:
    """Generate a non-extreme Rayleigh fading coefficient."""
    while True:
        coefficient = complex(
            (
                rng.standard_normal()
                + 1j * rng.standard_normal()
            )
            / np.sqrt(2)
        )

        if abs(coefficient) >= 0.25:
            return coefficient


def main() -> None:
    """Generate a balanced fault-classification dataset."""
    rng = np.random.default_rng(BASE_SEED)

    rows: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()

    modulation_names = list(MODULATIONS)

    for fault_type in FAULT_TYPES:
        for _ in range(SAMPLES_PER_FAULT):
            modulation = str(
                rng.choice(modulation_names)
            )

            snr_db = float(rng.uniform(25, 45))
            channel_coefficient = (
                generate_channel_coefficient(rng)
            )

            result = simulate_faulty_link_frame(
                fault_type=fault_type,
                modulation=modulation,
                snr_db=snr_db,
                channel_coefficient=channel_coefficient,
                rng=rng,
            )

            rows.append(
                {
                    "snr_db": snr_db,
                    "modulation": modulation,
                    "bits_per_symbol": MODULATIONS[
                        modulation
                    ][0],
                    "ber": result.ber,
                    "evm_percent": result.evm_percent,
                    "pilot_error_percent": (
                        result.pilot_error_percent
                    ),
                    "average_symbol_power": (
                        result.average_symbol_power
                    ),
                    "estimated_channel_magnitude": (
                        result.estimated_channel_magnitude
                    ),
                    # Stored for offline analysis only.
                    # The classifier will not use this field.
                    "channel_estimation_error_percent": (
                        result.channel_estimation_error_percent
                    ),
                    "true_channel_magnitude": abs(
                        channel_coefficient
                    ),
                    "fault_type": fault_type,
                }
            )

            label_counts[fault_type] += 1

        print(
            f"Generated {label_counts[fault_type]} "
            f"samples for {fault_type}"
        )

    rng.shuffle(rows)

    fieldnames = [
        "snr_db",
        "modulation",
        "bits_per_symbol",
        "ber",
        "evm_percent",
        "pilot_error_percent",
        "average_symbol_power",
        "estimated_channel_magnitude",
        "channel_estimation_error_percent",
        "true_channel_magnitude",
        "fault_type",
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
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

    print(f"\nDataset saved to {OUTPUT_PATH}")
    print(f"Total samples: {len(rows)}")
    print("Label distribution:")

    for fault_type in FAULT_TYPES:
        print(
            f"  {fault_type}: "
            f"{label_counts[fault_type]}"
        )


if __name__ == "__main__":
    main()
