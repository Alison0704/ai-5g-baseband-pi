"""Compare modulation-selection strategies under fading and interference."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from src.link_simulator import MODULATIONS, simulate_link_frame
from src.ml.modulation_selector import predict_modulation


FFT_SIZE = 64
PILOT_INDICES = (7, 21, 43, 57)

NUM_CHANNEL_CONDITIONS = 300
PROBE_FRAMES = 4
EVALUATION_FRAMES = 8
TARGET_BER = 0.01
BASE_SEED = 2026


def create_rng(
    condition_index: int,
    stream_index: int,
) -> np.random.Generator:
    """Create a reproducible independent random stream."""
    seed = np.random.SeedSequence(
        [BASE_SEED, condition_index, stream_index]
    )

    return np.random.default_rng(seed)


def generate_channel_coefficient(
    rng: np.random.Generator,
) -> complex:
    """Generate a Rayleigh fading coefficient without extreme fades."""
    while True:
        coefficient = complex(
            (
                rng.standard_normal()
                + 1j * rng.standard_normal()
            )
            / np.sqrt(2)
        )

        if abs(coefficient) >= 0.15:
            return coefficient


def generate_channel_condition(
    condition_index: int,
) -> dict[str, object]:
    """Generate one reproducible channel condition."""
    rng = create_rng(condition_index, stream_index=0)

    snr_db = float(rng.uniform(0, 30))
    channel_coefficient = generate_channel_coefficient(rng)

    interference_present = bool(rng.random() < 0.60)

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

    return {
        "snr_db": snr_db,
        "channel_coefficient": channel_coefficient,
        "interference_present": interference_present,
        "interference_to_signal_db": (
            interference_to_signal_db
        ),
        "interference_bin": interference_bin,
        "interference_frequency": interference_frequency,
        "interference_phase": interference_phase,
    }


def measure_probe(
    condition: dict[str, object],
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
            snr_db=float(condition["snr_db"]),
            channel_coefficient=complex(
                condition["channel_coefficient"]
            ),
            interference_to_signal_db=condition[
                "interference_to_signal_db"
            ],
            interference_frequency=float(
                condition["interference_frequency"]
            ),
            interference_phase_rad=float(
                condition["interference_phase"]
            ),
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


def rule_based_selection(snr_db: float) -> str:
    """Select modulation using fixed SNR thresholds."""
    if snr_db < 12:
        return "QPSK"

    if snr_db < 20:
        return "16QAM"

    return "64QAM"


def evaluate_modulation(
    modulation: str,
    condition: dict[str, object],
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Evaluate one modulation under one channel condition.

    Returns:
        Average BER.
        Successful-frame rate.
        Successfully delivered bits per frame.
    """
    ber_values: list[float] = []
    successful_frames = 0

    bits_per_symbol = MODULATIONS[modulation][0]
    data_subcarriers = FFT_SIZE - len(PILOT_INDICES)
    bits_per_frame = data_subcarriers * bits_per_symbol

    for _ in range(EVALUATION_FRAMES):
        result = simulate_link_frame(
            modulation=modulation,
            snr_db=float(condition["snr_db"]),
            channel_coefficient=complex(
                condition["channel_coefficient"]
            ),
            interference_to_signal_db=condition[
                "interference_to_signal_db"
            ],
            interference_frequency=float(
                condition["interference_frequency"]
            ),
            interference_phase_rad=float(
                condition["interference_phase"]
            ),
            rng=rng,
        )

        ber_values.append(result.ber)

        if result.ber <= TARGET_BER:
            successful_frames += 1

    success_rate = (
        successful_frames / EVALUATION_FRAMES
    )

    delivered_bits = success_rate * bits_per_frame

    return (
        float(np.mean(ber_values)),
        success_rate,
        delivered_bits,
    )


def main() -> None:
    """Run the full-channel strategy comparison."""
    output_path = Path(
        "results/full_channel_link_adaptation.csv"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, object]] = []

    strategy_totals = {
        "fixed_qpsk": {
            "ber": [],
            "success": [],
            "delivered_bits": [],
        },
        "rule_based": {
            "ber": [],
            "success": [],
            "delivered_bits": [],
        },
        "ml_based": {
            "ber": [],
            "success": [],
            "delivered_bits": [],
        },
    }

    ml_selection_counts: Counter[str] = Counter()

    evaluation_streams = {
        "QPSK": 10,
        "16QAM": 11,
        "64QAM": 12,
    }

    for condition_index in range(
        NUM_CHANNEL_CONDITIONS
    ):
        condition = generate_channel_condition(
            condition_index
        )

        probe = measure_probe(
            condition=condition,
            rng=create_rng(
                condition_index,
                stream_index=1,
            ),
        )

        selections = {
            "fixed_qpsk": "QPSK",
            "rule_based": rule_based_selection(
                float(condition["snr_db"])
            ),
            "ml_based": predict_modulation(
                snr_db=float(condition["snr_db"]),
                probe_evm_percent=probe[
                    "probe_evm_percent"
                ],
                probe_ber=probe["probe_ber"],
                packet_error_rate=probe[
                    "packet_error_rate"
                ],
                probe_pilot_error_percent=probe[
                    "probe_pilot_error_percent"
                ],
                estimated_channel_magnitude=probe[
                    "estimated_channel_magnitude"
                ],
            ),
        }

        ml_selection_counts[
            selections["ml_based"]
        ] += 1

        # Evaluate each modulation once. Strategies selecting the
        # same modulation receive identical measurements.
        evaluation_cache = {}

        for modulation in MODULATIONS:
            evaluation_cache[modulation] = (
                evaluate_modulation(
                    modulation=modulation,
                    condition=condition,
                    rng=create_rng(
                        condition_index,
                        evaluation_streams[modulation],
                    ),
                )
            )

        for strategy, modulation in selections.items():
            ber, success_rate, delivered_bits = (
                evaluation_cache[modulation]
            )

            strategy_totals[strategy]["ber"].append(ber)
            strategy_totals[strategy]["success"].append(
                success_rate
            )
            strategy_totals[strategy][
                "delivered_bits"
            ].append(delivered_bits)

            rows.append(
                {
                    "condition": condition_index,
                    "strategy": strategy,
                    "snr_db": condition["snr_db"],
                    **probe,
                    "selected_modulation": modulation,
                    "average_ber": ber,
                    "success_rate": success_rate,
                    "delivered_bits_per_frame": (
                        delivered_bits
                    ),
                    "true_channel_magnitude": abs(
                        complex(
                            condition[
                                "channel_coefficient"
                            ]
                        )
                    ),
                    "interference_present": int(
                        bool(
                            condition[
                                "interference_present"
                            ]
                        )
                    ),
                    "interference_to_signal_db": (
                        condition[
                            "interference_to_signal_db"
                        ]
                    ),
                    "interference_bin": condition[
                        "interference_bin"
                    ],
                }
            )

        if (condition_index + 1) % 50 == 0:
            print(
                f"Evaluated {condition_index + 1}/"
                f"{NUM_CHANNEL_CONDITIONS} conditions"
            )

    fieldnames = [
        "condition",
        "strategy",
        "snr_db",
        "probe_evm_percent",
        "probe_ber",
        "packet_error_rate",
        "probe_pilot_error_percent",
        "estimated_channel_magnitude",
        "selected_modulation",
        "average_ber",
        "success_rate",
        "delivered_bits_per_frame",
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

    print("\nStrategy comparison")
    print("-" * 72)
    print(
        f"{'Strategy':<16}"
        f"{'Average BER':>15}"
        f"{'Success rate':>18}"
        f"{'Delivered bits':>20}"
    )

    for strategy, metrics in strategy_totals.items():
        print(
            f"{strategy:<16}"
            f"{np.mean(metrics['ber']):>15.6f}"
            f"{np.mean(metrics['success']):>18.3f}"
            f"{np.mean(metrics['delivered_bits']):>20.2f}"
        )

    print("\nML selections:")

    for modulation in MODULATIONS:
        print(
            f"  {modulation}: "
            f"{ml_selection_counts[modulation]}"
        )

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
