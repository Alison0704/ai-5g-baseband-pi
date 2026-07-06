"""Generate portfolio-ready plots from saved experiment results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


def plot_modulation_ber() -> None:
    """Plot BER against SNR for each modulation."""
    input_path = RESULTS_DIR / "modulation_snr_comparison.csv"

    if not input_path.exists():
        print(f"Skipped missing file: {input_path}")
        return

    dataframe = pd.read_csv(input_path)

    plt.figure(figsize=(8, 5))

    for modulation, group in dataframe.groupby("modulation"):
        group = group.sort_values("snr_db")

        plt.semilogy(
            group["snr_db"],
            group["ber"].clip(lower=1e-5),
            marker="o",
            label=modulation,
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("BER Performance by Modulation")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    output_path = PLOTS_DIR / "ber_vs_snr.png"
    plt.savefig(output_path, dpi=180)
    plt.close()

    print(f"Generated {output_path}")


def plot_strategy_throughput() -> None:
    """Compare average delivered bits for each strategy."""
    input_path = RESULTS_DIR / "full_channel_link_adaptation.csv"

    if not input_path.exists():
        print(f"Skipped missing file: {input_path}")
        return

    dataframe = pd.read_csv(input_path)

    summary = (
        dataframe.groupby("strategy")[
            "delivered_bits_per_frame"
        ]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(7, 5))
    summary.plot(kind="bar")

    plt.xlabel("Selection strategy")
    plt.ylabel("Average delivered bits per frame")
    plt.title("Link-Adaptation Strategy Comparison")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = PLOTS_DIR / "strategy_throughput.png"
    plt.savefig(output_path, dpi=180)
    plt.close()

    print(f"Generated {output_path}")


def plot_pi_performance() -> None:
    """Plot Raspberry Pi baseband frame throughput."""
    candidates = [
        RESULTS_DIR / "system_benchmark_raspberry_pi_cached.csv",
        RESULTS_DIR / "system_benchmark_raspberry_pi.csv",
        RESULTS_DIR / "system_benchmark.csv",
    ]

    input_path = next(
        (path for path in candidates if path.exists()),
        None,
    )

    if input_path is None:
        print("Skipped missing Raspberry Pi benchmark")
        return

    dataframe = pd.read_csv(input_path)

    frame_rows = dataframe[
        dataframe["operation"].str.endswith(
            "_link_frame",
            na=False,
        )
    ].copy()

    frame_rows["modulation"] = (
        frame_rows["operation"]
        .str.replace("_link_frame", "", regex=False)
    )

    plt.figure(figsize=(7, 5))
    plt.bar(
        frame_rows["modulation"],
        frame_rows["frames_per_second"],
    )

    plt.xlabel("Modulation")
    plt.ylabel("Frames per second")
    plt.title("Raspberry Pi 4 Baseband Performance")
    plt.tight_layout()

    output_path = PLOTS_DIR / "pi_frame_throughput.png"
    plt.savefig(output_path, dpi=180)
    plt.close()

    print(f"Generated {output_path}")


def main() -> None:
    """Generate all available plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_modulation_ber()
    plot_strategy_throughput()
    plot_pi_performance()


if __name__ == "__main__":
    main()
