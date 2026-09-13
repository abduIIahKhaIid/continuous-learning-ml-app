from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_verified_metrics(records: list[Any]) -> dict[str, Any]:
    """Compute classification metrics from verified ground truth only."""
    if not records:
        raise ValueError("At least one verified prediction is required.")
    actual = np.asarray([record.actual_label for record in records])
    predicted = np.asarray([record.predicted_class for record in records])
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1_score": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).tolist(),
    }
    probabilities = [record.prediction_probability for record in records]
    if len(np.unique(actual)) == 2 and all(
        probability is not None for probability in probabilities
    ):
        metrics["roc_auc"] = float(roc_auc_score(actual, probabilities))
    return metrics


def classify_performance_drop(
    baseline: float,
    current: float,
    *,
    warning_drop: float,
    critical_drop: float,
) -> tuple[str, float]:
    drop = baseline - current
    if drop >= critical_drop:
        return "critical", drop
    if drop >= warning_drop:
        return "warning", drop
    return "stable", drop

