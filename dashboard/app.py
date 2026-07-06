"""Interactive dashboard for the AI-driven OFDM link simulator."""

import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import streamlit as st

from src.fault_injection import (
    FAULT_TYPES,
    simulate_faulty_link_frame,
)
from src.ml.failure_classifier import diagnose_fault


# Allow imports from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.link_simulator import MODULATIONS, simulate_link_frame
from src.ml.modulation_selector import predict_modulation


PROBE_FRAMES = 4
FFT_SIZE = 64


def create_channel_coefficient(
    magnitude: float,
    phase_degrees: float,
) -> complex:
    """Create a complex channel coefficient from polar parameters."""
    phase_radians = np.deg2rad(phase_degrees)

    return complex(
        magnitude * np.exp(1j * phase_radians)
    )


def measure_qpsk_probe(
    *,
    snr_db: float,
    channel_coefficient: complex,
    interference_to_signal_db: float | None,
    interference_frequency: float,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run several QPSK probes and calculate averaged features."""
    rng = np.random.default_rng(seed)

    rows: list[dict[str, float]] = []
    packet_errors = 0

    for frame_index in range(PROBE_FRAMES):
        result = simulate_link_frame(
            modulation="QPSK",
            snr_db=snr_db,
            channel_coefficient=channel_coefficient,
            interference_to_signal_db=(
                interference_to_signal_db
            ),
            interference_frequency=interference_frequency,
            rng=rng,
        )

        if result.ber > 0:
            packet_errors += 1

        rows.append(
            {
                "frame": frame_index + 1,
                "ber": result.ber,
                "evm_percent": result.evm_percent,
                "pilot_error_percent": (
                    result.pilot_error_percent
                ),
                "estimated_channel_magnitude": abs(
                    result.estimated_channel
                ),
            }
        )

    dataframe = pd.DataFrame(rows)

    features = {
        "snr_db": snr_db,
        "probe_evm_percent": float(
            dataframe["evm_percent"].mean()
        ),
        "probe_ber": float(
            dataframe["ber"].mean()
        ),
        "packet_error_rate": (
            packet_errors / PROBE_FRAMES
        ),
        "probe_pilot_error_percent": float(
            dataframe["pilot_error_percent"].mean()
        ),
        "estimated_channel_magnitude": float(
            dataframe[
                "estimated_channel_magnitude"
            ].mean()
        ),
    }

    return features, dataframe


st.set_page_config(
    page_title="AI 5G Baseband Simulator",
    page_icon="📡",
    layout="wide",
)

st.title("AI-Driven OFDM Link Simulator")

st.write(
    "Simulate an NR-inspired OFDM link and use the trained "
    "machine-learning model to select QPSK, 16-QAM, or 64-QAM."
)

with st.sidebar:
    st.header("Channel configuration")

    snr_db = st.slider(
        "SNR (dB)",
        min_value=0.0,
        max_value=30.0,
        value=20.0,
        step=0.5,
    )

    channel_magnitude = st.slider(
        "Channel magnitude",
        min_value=0.15,
        max_value=2.0,
        value=0.85,
        step=0.05,
    )

    channel_phase_degrees = st.slider(
        "Channel phase (degrees)",
        min_value=-180.0,
        max_value=180.0,
        value=20.0,
        step=5.0,
    )

    enable_interference = st.checkbox(
        "Enable tone interference",
        value=False,
    )
    st.divider()
    st.header("Fault diagnosis")

    enable_fault_diagnosis = st.checkbox(
        "Run fault diagnosis",
        value=False,
    )

    if enable_fault_diagnosis:
        injected_fault = st.selectbox(
            "Injected fault",
            options=list(FAULT_TYPES),
        )

        fault_modulation = st.selectbox(
            "Fault-test modulation",
            options=list(MODULATIONS),
            index=1,
        )

        fault_snr_db = st.slider(
            "Fault-test SNR (dB)",
            min_value=25.0,
            max_value=45.0,
            value=35.0,
            step=1.0,
        )
    else:
        injected_fault = "NONE"
        fault_modulation = "16QAM"
        fault_snr_db = 35.0

    if enable_interference:
        interference_to_signal_db = st.slider(
            "Interference-to-signal ratio (dB)",
            min_value=-30.0,
            max_value=0.0,
            value=-20.0,
            step=1.0,
        )

        interference_bin = st.slider(
            "Interference subcarrier",
            min_value=-32,
            max_value=31,
            value=10,
        )
    else:
        interference_to_signal_db = None
        interference_bin = 0

    manual_modulation = st.selectbox(
        "Manual modulation",
        options=list(MODULATIONS),
    )

    use_ml_recommendation = st.checkbox(
        "Transmit using ML recommendation",
        value=True,
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=2026,
        step=1,
    )

    run_simulation = st.button(
        "Run simulation",
        type="primary",
        use_container_width=True,
    )


if run_simulation:
    channel_coefficient = create_channel_coefficient(
        magnitude=channel_magnitude,
        phase_degrees=channel_phase_degrees,
    )

    interference_frequency = (
        interference_bin / FFT_SIZE
    )

    probe_features, probe_dataframe = measure_qpsk_probe(
        snr_db=snr_db,
        channel_coefficient=channel_coefficient,
        interference_to_signal_db=(
            interference_to_signal_db
        ),
        interference_frequency=interference_frequency,
        seed=int(seed),
    )

    try:
        recommended_modulation = predict_modulation(
            snr_db=probe_features["snr_db"],
            probe_evm_percent=probe_features[
                "probe_evm_percent"
            ],
            probe_ber=probe_features["probe_ber"],
            packet_error_rate=probe_features[
                "packet_error_rate"
            ],
            probe_pilot_error_percent=probe_features[
                "probe_pilot_error_percent"
            ],
            estimated_channel_magnitude=probe_features[
                "estimated_channel_magnitude"
            ],
        )
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    selected_modulation = (
        recommended_modulation
        if use_ml_recommendation
        else manual_modulation
    )

    result = simulate_link_frame(
        modulation=selected_modulation,
        snr_db=snr_db,
        channel_coefficient=channel_coefficient,
        interference_to_signal_db=(
            interference_to_signal_db
        ),
        interference_frequency=interference_frequency,
        rng=np.random.default_rng(int(seed) + 1),
    )

    st.subheader("Link-adaptation decision")

    decision_columns = st.columns(3)

    decision_columns[0].metric(
        "ML recommendation",
        recommended_modulation,
    )

    decision_columns[1].metric(
        "Transmitted modulation",
        selected_modulation,
    )

    decision_columns[2].metric(
        "Bits per symbol",
        MODULATIONS[selected_modulation][0],
    )

    st.subheader("Receiver measurements")

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "BER",
        f"{result.ber:.5f}",
    )

    metric_columns[1].metric(
        "EVM",
        f"{result.evm_percent:.2f}%",
    )

    metric_columns[2].metric(
        "Pilot error",
        f"{result.pilot_error_percent:.2f}%",
    )

    metric_columns[3].metric(
        "Estimated channel",
        f"{abs(result.estimated_channel):.3f}",
    )

    if result.ber <= 0.01:
        st.success(
            "Frame satisfies the 1% BER target."
        )
    else:
        st.warning(
            "Frame exceeds the 1% BER target."
        )

    st.subheader("QPSK probe measurements")

    st.dataframe(
        probe_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    feature_dataframe = pd.DataFrame(
        [
            {
                "feature": feature,
                "value": value,
            }
            for feature, value in probe_features.items()
        ]
    )

    st.subheader("ML model inputs")

    st.dataframe(
        feature_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Channel comparison")

    channel_dataframe = pd.DataFrame(
        [
            {
                "measurement": "True magnitude",
                "value": abs(result.true_channel),
            },
            {
                "measurement": "Estimated magnitude",
                "value": abs(result.estimated_channel),
            },
            {
                "measurement": "True phase (degrees)",
                "value": np.angle(
                    result.true_channel,
                    deg=True,
                ),
            },
            {
                "measurement": "Estimated phase (degrees)",
                "value": np.angle(
                    result.estimated_channel,
                    deg=True,
                ),
            },
        ]
    )

    st.dataframe(
        channel_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    if enable_fault_diagnosis:
        st.divider()
        st.subheader("ML-assisted fault diagnosis")

        fault_result = simulate_faulty_link_frame(
            fault_type=injected_fault,
            modulation=fault_modulation,
            snr_db=fault_snr_db,
            channel_coefficient=channel_coefficient,
            rng=np.random.default_rng(int(seed) + 100),
        )

        try:
            diagnosis = diagnose_fault(
                snr_db=fault_snr_db,
                modulation=fault_result.modulation,
                bits_per_symbol=MODULATIONS[
                    fault_result.modulation
                ][0],
                ber=fault_result.ber,
                evm_percent=fault_result.evm_percent,
                pilot_error_percent=(
                    fault_result.pilot_error_percent
                ),
                average_symbol_power=(
                    fault_result.average_symbol_power
                ),
                estimated_channel_magnitude=(
                    fault_result.estimated_channel_magnitude
                ),
            )
        except FileNotFoundError as error:
            st.error(str(error))
            st.stop()

        diagnosed_fault = str(diagnosis["fault_type"])
        diagnosis_confidence = float(
            cast(float, diagnosis["confidence"])
        )
        diagnosis_probabilities = cast(
            dict[str, float],
            diagnosis["probabilities"],
        )

        diagnosis_columns = st.columns(3)

        diagnosis_columns[0].metric(
            "Injected condition",
            injected_fault,
        )

        diagnosis_columns[1].metric(
            "Predicted condition",
            diagnosed_fault,
        )

        diagnosis_columns[2].metric(
            "Confidence",
            f"{diagnosis_confidence * 100:.1f}%",
        )

        if diagnosed_fault == injected_fault:
            st.success(
                "The classifier correctly identified the "
                "injected condition."
            )
        else:
            st.warning(
                "The predicted condition differs from the "
                "injected condition."
            )

        fault_metrics = pd.DataFrame(
            [
                {
                    "measurement": "BER",
                    "value": fault_result.ber,
                },
                {
                    "measurement": "EVM (%)",
                    "value": fault_result.evm_percent,
                },
                {
                    "measurement": "Pilot error (%)",
                    "value": (
                        fault_result.pilot_error_percent
                    ),
                },
                {
                    "measurement": "Average symbol power",
                    "value": (
                        fault_result.average_symbol_power
                    ),
                },
                {
                    "measurement": (
                        "Estimated channel magnitude"
                    ),
                    "value": (
                        fault_result
                        .estimated_channel_magnitude
                    ),
                },
            ]
        )

        st.write("Diagnostic measurements")

        st.dataframe(
            fault_metrics,
            use_container_width=True,
            hide_index=True,
        )

        probability_dataframe = pd.DataFrame(
            [
                {
                    "fault_type": fault_type,
                    "probability_percent": (
                        probability * 100
                    ),
                }
                for fault_type, probability
                in diagnosis_probabilities.items()
            ]
        ).sort_values(
            "probability_percent",
            ascending=False,
        )

        st.write("Classifier probabilities")

        st.dataframe(
            probability_dataframe,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            probability_dataframe.set_index(
                "fault_type"
            )["probability_percent"]
        )
else:
    st.info(
        "Configure the channel in the sidebar and select "
        "Run simulation."
    )
