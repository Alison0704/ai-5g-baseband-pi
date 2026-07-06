"""Modulation-selection utilities."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import pandas as pd

import joblib
import numpy as np

MODULATION_ORDER = ("QPSK", "16QAM", "64QAM")
DEFAULT_MODEL_PATH = Path("models/link_adapter.joblib")


def select_best_modulation_from_ber(
    ber_by_modulation: Mapping[str, float],
    target_ber: float = 0.01,
) -> str:
    """
    Select the highest-capacity modulation meeting the BER target.

    Falls back to QPSK when none satisfies the target.
    """
    if not 0 <= target_ber <= 1:
        raise ValueError("Target BER must be between 0 and 1.")

    missing = set(MODULATION_ORDER) - set(ber_by_modulation)

    if missing:
        raise ValueError(
            f"Missing BER values for: {sorted(missing)}"
        )

    for modulation in reversed(MODULATION_ORDER):
        ber = ber_by_modulation[modulation]

        if not 0 <= ber <= 1:
            raise ValueError(
                f"Invalid BER for {modulation}: {ber}"
            )

        if ber <= target_ber:
            return modulation

    return "QPSK"


def load_link_adapter(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Load a trusted link-adaptation model artifact."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run python -m src.ml.train_link_adapter first."
        )

    artifact = joblib.load(model_path)

    required_keys = {"model", "features", "labels"}

    if not required_keys.issubset(artifact):
        raise ValueError("Invalid model artifact.")

    return artifact


def predict_modulation(
    snr_db: float,
    probe_evm_percent: float,
    probe_ber: float,
    packet_error_rate: float,
    probe_pilot_error_percent: float,
    estimated_channel_magnitude: float,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> str:
    """Predict the best modulation from receiver measurements."""
    if probe_evm_percent < 0:
        raise ValueError("EVM cannot be negative.")

    if not 0 <= probe_ber <= 1:
        raise ValueError("Probe BER must be between 0 and 1.")

    if not 0 <= packet_error_rate <= 1:
        raise ValueError(
            "Packet-error rate must be between 0 and 1."
        )

    if probe_pilot_error_percent < 0:
        raise ValueError(
            "Pilot error cannot be negative."
        )

    if estimated_channel_magnitude < 0:
        raise ValueError(
            "Estimated channel magnitude cannot be negative."
        )

    artifact = load_link_adapter(model_path)

    feature_values = {
        "snr_db": snr_db,
        "probe_evm_percent": probe_evm_percent,
        "probe_ber": probe_ber,
        "packet_error_rate": packet_error_rate,
        "probe_pilot_error_percent": (
            probe_pilot_error_percent
        ),
        "estimated_channel_magnitude": (
            estimated_channel_magnitude
        ),
    }

    feature_order = artifact["features"]

    model_input = pd.DataFrame(
        [[feature_values[name] for name in feature_order]],
        columns=feature_order,
        dtype=np.float64,
    )

    prediction = str(
        artifact["model"].predict(model_input)[0]
    )

    if prediction not in MODULATION_ORDER:
        raise ValueError(
            f"Model returned unknown modulation: {prediction}"
        )

    return prediction