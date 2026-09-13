from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None
    confusion_matrix: list[list[int]]


def evaluate_model(
    pipeline: Pipeline,
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> EvaluationMetrics:
    predictions = pipeline.predict(features)
    roc_auc: float | None = None
    if np.unique(labels).size == 2 and hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(features)[:, 1]
            roc_auc = float(roc_auc_score(labels, probabilities))
        except ValueError:
            roc_auc = None

    return EvaluationMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        f1_score=float(f1_score(labels, predictions, zero_division=0)),
        roc_auc=roc_auc,
        confusion_matrix=confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    )
