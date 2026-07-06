"""Benchmark baseband and ML performance on the current computer."""

import csv
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil

from src.link_simulator import simulate_link_frame
from src.ml.modulation_selector import predict_modulation


ITERATIONS = 300
OUTPUT_PATH = Path("results/system_benchmark.csv")


def benchmark_link(modulation: str) -> dict[str, float]:
    """Benchmark one complete OFDM frame."""
    rng = np.random.default_rng(2026)
    latencies = []

    for _ in range(ITERATIONS):
        start = perf_counter()

        simulate_link_frame(
            modulation=modulation,
            snr_db=20.0,
            channel_coefficient=0.8 + 0.3j,
            interference_to_signal_db=-20.0,
            rng=rng,
        )

        latencies.append(
            (perf_counter() - start) * 1000
        )

    values = np.asarray(latencies)

    return {
        "mean_latency_ms": float(values.mean()),
        "minimum_latency_ms": float(values.min()),
        "maximum_latency_ms": float(values.max()),
        "frames_per_second": float(
            1000 / values.mean()
        ),
    }


def benchmark_ml() -> dict[str, float]:
    """Benchmark modulation-selection inference."""
    latencies = []

    for _ in range(ITERATIONS):
        start = perf_counter()

        predict_modulation(
            snr_db=20.0,
            probe_evm_percent=15.0,
            probe_ber=0.0,
            packet_error_rate=0.0,
            probe_pilot_error_percent=8.0,
            estimated_channel_magnitude=0.85,
        )

        latencies.append(
            (perf_counter() - start) * 1000
        )

    values = np.asarray(latencies)

    return {
        "mean_latency_ms": float(values.mean()),
        "minimum_latency_ms": float(values.min()),
        "maximum_latency_ms": float(values.max()),
        "inferences_per_second": float(
            1000 / values.mean()
        ),
    }


def main() -> None:
    """Run and save all system benchmarks."""
    process = psutil.Process()

    rows = []

    print("System information")
    print("-" * 50)
    print("Platform:", platform.platform())
    print("Machine: ", platform.machine())
    print("Python:  ", platform.python_version())
    print("CPUs:    ", psutil.cpu_count(logical=True))
    print(
        "Memory:  ",
        f"{psutil.virtual_memory().total / 1024**3:.2f} GB",
    )

    for modulation in ("QPSK", "16QAM", "64QAM"):
        result = benchmark_link(modulation)

        rows.append(
            {
                "operation": f"{modulation}_link_frame",
                **result,
            }
        )

        print(
            f"\n{modulation}: "
            f"{result['mean_latency_ms']:.4f} ms, "
            f"{result['frames_per_second']:.2f} frames/s"
        )

    ml_result = benchmark_ml()

    rows.append(
        {
            "operation": "ml_modulation_selection",
            **ml_result,
        }
    )

    print(
        "\nML inference: "
        f"{ml_result['mean_latency_ms']:.4f} ms, "
        f"{ml_result['inferences_per_second']:.2f} inferences/s"
    )

    print(
        "Process memory:",
        f"{process.memory_info().rss / 1024**2:.2f} MB",
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
        }
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

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
