"""Train the ML-based modulation selector."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


DATASET_PATH = Path("datasets/link_adaptation.csv")
MODEL_PATH = Path("models/link_adapter.joblib")
REPORT_PATH = Path("results/link_adapter_report.json")

FEATURE_COLUMNS = [
    "snr_db",
    "probe_evm_percent",
    "probe_ber",
    "packet_error_rate",
    "probe_pilot_error_percent",
    "estimated_channel_magnitude",
]

TARGET_COLUMN = "selected_modulation"
LABELS = ["QPSK", "16QAM", "64QAM"]


def load_dataset(
    dataset_path: Path = DATASET_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the link-adaptation dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run python -m scripts.generate_dataset first."
        )

    dataframe = pd.read_csv(dataset_path)

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    if dataframe[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Feature columns contain missing values.")

    if dataframe[TARGET_COLUMN].isnull().any():
        raise ValueError("Target column contains missing values.")

    features = dataframe[FEATURE_COLUMNS]
    target = dataframe[TARGET_COLUMN]

    return features, target


def main() -> None:
    """Train, evaluate, and save the decision-tree model."""
    features, target = load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS,
    )

    artifact = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "labels": LABELS,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, MODEL_PATH)

    result = {
        "accuracy": accuracy,
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "features": FEATURE_COLUMNS,
        "labels": LABELS,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples:     {len(x_test)}")
    print(f"Accuracy:         {accuracy:.3f}")
    print()
    print("Confusion matrix:")
    print(f"Labels: {LABELS}")
    print(matrix)
    print()
    print(f"Model saved to {MODEL_PATH}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
