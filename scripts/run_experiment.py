"""Compare QPSK, 16-QAM, and 64-QAM across SNR values."""

import csv
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from src.channel.awgn import add_awgn
from src.metrics.ber import calculate_ber
from src.metrics.evm import calculate_evm_rms, evm_to_percent
from src.receiver.demapper import (
    qam16_demodulate,
    qam64_demodulate,
    qpsk_demodulate,
)
from src.receiver.ofdm_rx import ofdm_demodulate
from src.transmitter.modulator import (
    qam16_modulate,
    qam64_modulate,
    qpsk_modulate,
)
from src.transmitter.ofdm_tx import ofdm_modulate


FFT_SIZE = 64
CP_LENGTH = 16
FRAMES_PER_POINT = 200
SNR_VALUES_DB = [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

Modulator = Callable[[NDArray[np.int8]], NDArray[np.complex128]]
Demodulator = Callable[[NDArray[np.complex128]], NDArray[np.int8]]


MODULATIONS: dict[str, dict[str, object]] = {
    "QPSK": {
        "bits_per_symbol": 2,
        "modulate": qpsk_modulate,
        "demodulate": qpsk_demodulate,
    },
    "16QAM": {
        "bits_per_symbol": 4,
        "modulate": qam16_modulate,
        "demodulate": qam16_demodulate,
    },
    "64QAM": {
        "bits_per_symbol": 6,
        "modulate": qam64_modulate,
        "demodulate": qam64_demodulate,
    },
}


def run_snr_point(
    snr_db: float,
    bits_per_symbol: int,
    modulate: Modulator,
    demodulate: Demodulator,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Measure average BER and EVM for one configuration."""
    total_bit_errors = 0
    total_bits = 0
    evm_values: list[float] = []

    bits_per_frame = FFT_SIZE * bits_per_symbol

    for _ in range(FRAMES_PER_POINT):
        transmitted_bits = rng.integers(
            0,
            2,
            size=bits_per_frame,
            dtype=np.int8,
        )

        transmitted_symbols = modulate(transmitted_bits)

        transmitted_signal = ofdm_modulate(
            transmitted_symbols,
            fft_size=FFT_SIZE,
            cp_length=CP_LENGTH,
        )

        received_signal = add_awgn(
            transmitted_signal,
            snr_db=snr_db,
            rng=rng,
        )

        received_symbols = ofdm_demodulate(
            received_signal,
            fft_size=FFT_SIZE,
            cp_length=CP_LENGTH,
            num_data_symbols=FFT_SIZE,
        )

        received_bits = demodulate(received_symbols)

        total_bit_errors += int(
            np.count_nonzero(transmitted_bits != received_bits)
        )
        total_bits += transmitted_bits.size

        evm_values.append(
            calculate_evm_rms(
                transmitted_symbols,
                received_symbols,
            )
        )

    average_ber = total_bit_errors / total_bits
    average_evm = float(np.mean(evm_values))

    return average_ber, average_evm


def main() -> None:
    """Run the comparison and save a combined CSV dataset."""
    rng = np.random.default_rng(12345)

    output_path = Path("results/modulation_snr_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    print(
        f"{'Modulation':>10} "
        f"{'SNR (dB)':>9} "
        f"{'BER':>12} "
        f"{'EVM (%)':>12}"
    )
    print("-" * 48)

    for modulation_name, configuration in MODULATIONS.items():
        bits_per_symbol = int(configuration["bits_per_symbol"])
        modulate = configuration["modulate"]
        demodulate = configuration["demodulate"]

        assert callable(modulate)
        assert callable(demodulate)

        for snr_db in SNR_VALUES_DB:
            ber, evm = run_snr_point(
                snr_db=snr_db,
                bits_per_symbol=bits_per_symbol,
                modulate=modulate,
                demodulate=demodulate,
                rng=rng,
            )

            evm_percent = evm_to_percent(evm)

            rows.append(
                {
                    "modulation": modulation_name,
                    "snr_db": snr_db,
                    "bits_per_symbol": bits_per_symbol,
                    "bits_per_ofdm_symbol": FFT_SIZE * bits_per_symbol,
                    "ber": ber,
                    "evm_percent": evm_percent,
                }
            )

            print(
                f"{modulation_name:>10} "
                f"{snr_db:>9.1f} "
                f"{ber:>12.6f} "
                f"{evm_percent:>12.3f}"
            )

        print()

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "modulation",
                "snr_db",
                "bits_per_symbol",
                "bits_per_ofdm_symbol",
                "ber",
                "evm_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
