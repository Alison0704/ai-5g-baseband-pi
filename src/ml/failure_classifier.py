"""Train and run the OFDM fault-diagnosis classifier."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.fault_injection import FAULT_TYPES
from src.link_simulator import MODULATIONS


DATASET_PATH = Path("datasets/fault_diagnosis.csv")
MODEL_PATH = Path("models/failure_classifier.joblib")
REPORT_PATH = Path("results/failure_classifier_report.json")

FEATURE_COLUMNS = [
    "snr_db",
    "modulation",
    "bits_per_symbol",
    "ber",
    "evm_percent",
    "pilot_error_percent",
    "average_symbol_power",
    "estimated_channel_magnitude",
]

CATEGORICAL_FEATURES = ["modulation"]

NUMERIC_FEATURES = [
    "snr_db",
    "bits_per_symbol",
    "ber",
    "evm_percent",
    "pilot_error_percent",
    "average_symbol_power",
    "estimated_channel_magnitude",
]

TARGET_COLUMN = "fault_type"


def load_dataset(
    dataset_path: Path = DATASET_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the fault-diagnosis dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run python -m scripts.generate_fault_dataset first."
        )

    dataframe = pd.read_csv(dataset_path)

    required_columns = set(
        FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError(
            "Fault-classification features contain missing values."
        )

    unknown_labels = set(
        dataframe[TARGET_COLUMN].unique()
    ) - set(FAULT_TYPES)

    if unknown_labels:
        raise ValueError(
            f"Unknown fault labels: {sorted(unknown_labels)}"
        )

    return (
        dataframe[FEATURE_COLUMNS],
        dataframe[TARGET_COLUMN],
    )


def create_model() -> Pipeline:
    """Create the preprocessing and classification pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def load_failure_classifier(
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    """Load a trusted saved classifier artifact."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run python -m src.ml.failure_classifier first."
        )

    artifact = joblib.load(model_path)

    required_keys = {"model", "features", "labels"}

    if not required_keys.issubset(artifact):
        raise ValueError(
            "Invalid failure-classifier artifact."
        )

    return artifact


def diagnose_fault(
    *,
    snr_db: float,
    modulation: str,
    bits_per_symbol: int,
    ber: float,
    evm_percent: float,
    pilot_error_percent: float,
    average_symbol_power: float,
    estimated_channel_magnitude: float,
    model_path: Path = MODEL_PATH,
) -> dict[str, object]:
    """Predict the likely fault and return its confidence."""
    if modulation not in MODULATIONS:
        raise ValueError(
            f"Unsupported modulation: {modulation}"
        )

    expected_bits_per_symbol = MODULATIONS[
        modulation
    ][0]

    if bits_per_symbol != expected_bits_per_symbol:
        raise ValueError(
            f"{modulation} requires "
            f"{expected_bits_per_symbol} bits per symbol."
        )

    if not 0 <= ber <= 1:
        raise ValueError(
            "BER must be between 0 and 1."
        )

    if evm_percent < 0:
        raise ValueError("EVM cannot be negative.")

    if pilot_error_percent < 0:
        raise ValueError(
            "Pilot error cannot be negative."
        )

    if average_symbol_power < 0:
        raise ValueError(
            "Average symbol power cannot be negative."
        )

    if estimated_channel_magnitude < 0:
        raise ValueError(
            "Estimated channel magnitude cannot be negative."
        )

    artifact = load_failure_classifier(model_path)

    feature_values = {
        "snr_db": snr_db,
        "modulation": modulation,
        "bits_per_symbol": bits_per_symbol,
        "ber": ber,
        "evm_percent": evm_percent,
        "pilot_error_percent": pilot_error_percent,
        "average_symbol_power": average_symbol_power,
        "estimated_channel_magnitude": (
            estimated_channel_magnitude
        ),
    }

    model_input = pd.DataFrame(
        [[feature_values[name] for name in artifact["features"]]],
        columns=artifact["features"],
    )

    model = artifact["model"]

    prediction = str(
        model.predict(model_input)[0]
    )

    probabilities = model.predict_proba(
        model_input
    )[0]

    classes = model.named_steps[
        "classifier"
    ].classes_

    probability_by_class = {
        str(label): float(probability)
        for label, probability in zip(
            classes,
            probabilities,
        )
    }

    return {
        "fault_type": prediction,
        "confidence": probability_by_class[prediction],
        "probabilities": probability_by_class,
    }


def main() -> None:
    """Train, evaluate, and save the classifier."""
    features, target = load_dataset()

    x_train, x_test, y_train, y_test = (
        train_test_split(
            features,
            target,
            test_size=0.20,
            random_state=42,
            stratify=target,
        )
    )

    model = create_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=list(FAULT_TYPES),
    )

    report = classification_report(
        y_test,
        predictions,
        labels=list(FAULT_TYPES),
        output_dict=True,
        zero_division=0,
    )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    transformed_feature_names = (
        preprocessor.get_feature_names_out()
    )

    feature_importances = sorted(
        zip(
            transformed_feature_names,
            classifier.feature_importances_,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    artifact = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "labels": list(FAULT_TYPES),
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    result = {
        "accuracy": accuracy,
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "features": FEATURE_COLUMNS,
        "labels": list(FAULT_TYPES),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "feature_importances": [
            {
                "feature": str(name),
                "importance": float(importance),
            }
            for name, importance in feature_importances
        ],
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples:     {len(x_test)}")
    print(f"Accuracy:         {accuracy:.3f}")

    print("\nConfusion matrix:")
    print(f"Labels: {list(FAULT_TYPES)}")
    print(matrix)

    print("\nMost important features:")

    for name, importance in feature_importances[:8]:
        print(
            f"  {name}: {importance:.4f}"
        )

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
