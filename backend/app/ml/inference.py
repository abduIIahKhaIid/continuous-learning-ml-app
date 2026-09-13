import logging
from dataclasses import dataclass

import numpy as np

from app.ml.model_loader import LoadedModel

logger = logging.getLogger(__name__)


class PredictionExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class InferenceResult:
    predicted_class: int
    probability: float | None
    model_version: str


def run_inference(
    loaded_model: LoadedModel,
    *,
    feature_1: float,
    feature_2: float,
    feature_3: float,
) -> InferenceResult:
    features = np.asarray(
        [[feature_1, feature_2, feature_3]], dtype=np.float64
    )
    try:
        predicted_class = int(loaded_model.pipeline.predict(features)[0])
    except Exception as error:
        raise PredictionExecutionError("Model prediction failed.") from error
    if predicted_class not in {0, 1}:
        raise PredictionExecutionError(
            "Binary classifier returned an unsupported class."
        )

    probability: float | None = None
    if hasattr(loaded_model.pipeline, "predict_proba"):
        try:
            classes = list(loaded_model.pipeline.classes_)
            if 1 in classes:
                positive_class_index = classes.index(1)
                probability = float(
                    loaded_model.pipeline.predict_proba(features)[
                        0, positive_class_index
                    ]
                )
        except Exception:
            logger.warning(
                "Probability is unavailable for model %s.",
                loaded_model.descriptor.model_version,
                exc_info=True,
            )
            probability = None

    return InferenceResult(
        predicted_class=predicted_class,
        probability=probability,
        model_version=loaded_model.descriptor.model_version,
    )
