"""Run an end-to-end baseband, ML, and fault-diagnosis demo."""

import numpy as np

from src.fault_injection import simulate_faulty_link_frame
from src.link_simulator import MODULATIONS, simulate_link_frame
from src.ml.failure_classifier import diagnose_fault
from src.ml.modulation_selector import predict_modulation


PROBE_FRAMES = 4


def measure_probe(
    *,
    snr_db: float,
    channel_coefficient: complex,
    interference_to_signal_db: float | None,
    interference_frequency: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Collect QPSK probe measurements for link adaptation."""
    ber_values = []
    evm_values = []
    pilot_error_values = []
    channel_magnitudes = []
    packet_errors = 0

    for _ in range(PROBE_FRAMES):
        result = simulate_link_frame(
            modulation="QPSK",
            snr_db=snr_db,
            channel_coefficient=channel_coefficient,
            interference_to_signal_db=interference_to_signal_db,
            interference_frequency=interference_frequency,
            rng=rng,
        )

        ber_values.append(result.ber)
        evm_values.append(result.evm_percent)
        pilot_error_values.append(result.pilot_error_percent)
        channel_magnitudes.append(abs(result.estimated_channel))

        if result.ber > 0:
            packet_errors += 1

    return {
        "snr_db": snr_db,
        "probe_evm_percent": float(np.mean(evm_values)),
        "probe_ber": float(np.mean(ber_values)),
        "packet_error_rate": packet_errors / PROBE_FRAMES,
        "probe_pilot_error_percent": float(
            np.mean(pilot_error_values)
        ),
        "estimated_channel_magnitude": float(
            np.mean(channel_magnitudes)
        ),
    }


def main() -> None:
    """Run the complete demonstration."""
    rng = np.random.default_rng(2026)

    snr_db = 20.0
    channel_coefficient = 0.8 + 0.3j
    interference_to_signal_db = -20.0
    interference_frequency = 10 / 64

    print("AI-Driven OFDM Baseband Demo")
    print("=" * 36)

    probe = measure_probe(
        snr_db=snr_db,
        channel_coefficient=channel_coefficient,
        interference_to_signal_db=interference_to_signal_db,
        interference_frequency=interference_frequency,
        rng=rng,
    )

    selected_modulation = predict_modulation(**probe)

    link_result = simulate_link_frame(
        modulation=selected_modulation,
        snr_db=snr_db,
        channel_coefficient=channel_coefficient,
        interference_to_signal_db=interference_to_signal_db,
        interference_frequency=interference_frequency,
        rng=rng,
    )

    print("\nLink adaptation")
    print("-" * 36)
    print(f"SNR:                    {snr_db:.1f} dB")
    print(
        f"Probe EVM:              "
        f"{probe['probe_evm_percent']:.2f}%"
    )
    print(
        f"Probe pilot error:      "
        f"{probe['probe_pilot_error_percent']:.2f}%"
    )
    print(f"Selected modulation:    {selected_modulation}")
    print(f"Received BER:           {link_result.ber:.5f}")
    print(f"Received EVM:           {link_result.evm_percent:.2f}%")
    print(
        f"BER target:             "
        f"{'PASS' if link_result.ber <= 0.01 else 'FAIL'}"
    )

    injected_fault = "PILOT_CORRUPTION"

    fault_result = simulate_faulty_link_frame(
        fault_type=injected_fault,
        modulation="16QAM",
        snr_db=35.0,
        channel_coefficient=channel_coefficient,
        rng=np.random.default_rng(123),
    )

    diagnosis = diagnose_fault(
        snr_db=35.0,
        modulation=fault_result.modulation,
        bits_per_symbol=MODULATIONS[
            fault_result.modulation
        ][0],
        ber=fault_result.ber,
        evm_percent=fault_result.evm_percent,
        pilot_error_percent=fault_result.pilot_error_percent,
        average_symbol_power=fault_result.average_symbol_power,
        estimated_channel_magnitude=(
            fault_result.estimated_channel_magnitude
        ),
    )

    print("\nFault diagnosis")
    print("-" * 36)
    print(f"Injected fault:         {injected_fault}")
    print(f"Predicted fault:        {diagnosis['fault_type']}")
    print(
        f"Confidence:             "
        f"{diagnosis['confidence'] * 100:.1f}%"
    )
    print(
        f"Diagnosis result:       "
        f"{'PASS' if diagnosis['fault_type'] == injected_fault else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
