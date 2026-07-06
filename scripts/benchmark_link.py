"""Benchmark the end-to-end wireless link simulator."""

import csv
from pathlib import Path

import numpy as np

from src.link_simulator import simulate_link_frame
from src.metrics.profiler import profile_function


ITERATIONS = 200
MODULATIONS = ("QPSK", "16QAM", "64QAM")


def main() -> None:
    """Benchmark each supported modulation."""
    output_path = Path(
        "results/link_benchmark.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print(
        f"{'Modulation':<12}"
        f"{'Mean (ms)':>12}"
        f"{'Min (ms)':>12}"
        f"{'Max (ms)':>12}"
        f"{'Frames/s':>12}"
    )

    print("-" * 60)

    for modulation in MODULATIONS:
        rng = np.random.default_rng(123)

        result = profile_function(
            simulate_link_frame,
            iterations=ITERATIONS,
            warmup_iterations=10,
            modulation=modulation,
            snr_db=25.0,
            channel_coefficient=0.8 + 0.3j,
            interference_to_signal_db=-20,
            rng=rng,
        )

        rows.append(
            {
                "modulation": modulation,
                "iterations": result.iterations,
                "mean_latency_ms": (
                    result.mean_latency_ms
                ),
                "minimum_latency_ms": (
                    result.minimum_latency_ms
                ),
                "maximum_latency_ms": (
                    result.maximum_latency_ms
                ),
                "standard_deviation_ms": (
                    result.standard_deviation_ms
                ),
                "frames_per_second": (
                    result.frames_per_second
                ),
            }
        )

        print(
            f"{modulation:<12}"
            f"{result.mean_latency_ms:>12.4f}"
            f"{result.minimum_latency_ms:>12.4f}"
            f"{result.maximum_latency_ms:>12.4f}"
            f"{result.frames_per_second:>12.2f}"
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
