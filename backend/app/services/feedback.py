import logging
import math

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.prediction import Prediction
from app.repositories.data import DataRepository
from app.repositories.prediction import PredictionRepository
from app.schemas.prediction import (
    PredictionFeedbackRequest,
    PredictionFeedbackResponse,
    PredictionFeedbackStatus,
    PredictionFeedbackSummary,
)

logger = logging.getLogger(__name__)


class FeedbackNotFoundError(Exception):
    pass


class FeedbackAlreadySubmittedError(Exception):
    pass


class InvalidPredictionDataError(ValueError):
    pass


class FeedbackDatabaseError(RuntimeError):
    pass


class FeedbackService:
    """Collect immutable ground truth and create its training-data lineage."""

    def __init__(
        self,
        *,
        session: Session,
        prediction_repository: PredictionRepository,
        data_repository: DataRepository,
    ) -> None:
        self._session = session
        self._predictions = prediction_repository
        self._training_data = data_repository

    def submit(
        self,
        prediction_id: int,
        payload: PredictionFeedbackRequest,
    ) -> PredictionFeedbackResponse:
        if self._session.in_transaction():
            self._session.rollback()
        try:
            with self._session.begin():
                prediction = self._predictions.get_for_feedback(prediction_id)
                if prediction is None:
                    raise FeedbackNotFoundError(prediction_id)

                existing_sample = (
                    self._training_data.get_by_source_prediction_id(
                        prediction_id
                    )
                )
                if (
                    prediction.feedback_received
                    or prediction.actual_label is not None
                    or existing_sample is not None
                ):
                    raise FeedbackAlreadySubmittedError(prediction_id)

                feature_values = (
                    prediction.feature_1,
                    prediction.feature_2,
                    prediction.feature_3,
                )
                if not all(
                    value is not None and math.isfinite(value)
                    for value in feature_values
                ):
                    raise InvalidPredictionDataError(
                        "Prediction features are missing or non-finite."
                    )

                actual_label = payload.actual_label
                self._predictions.set_feedback(
                    prediction, actual_label=actual_label
                )
                self._training_data.create_verified(
                    feature_1=prediction.feature_1,
                    feature_2=prediction.feature_2,
                    feature_3=prediction.feature_3,
                    actual_label=actual_label,
                    source_prediction_id=prediction.id,
                )
                self._session.flush()
        except (
            FeedbackNotFoundError,
            FeedbackAlreadySubmittedError,
            InvalidPredictionDataError,
        ):
            raise
        except IntegrityError as error:
            logger.info(
                "Duplicate feedback prevented for prediction %s.",
                prediction_id,
            )
            raise FeedbackAlreadySubmittedError(prediction_id) from error
        except SQLAlchemyError as error:
            logger.exception(
                "Feedback transaction failed for prediction %s.",
                prediction_id,
            )
            raise FeedbackDatabaseError(
                "Ground-truth feedback could not be stored."
            ) from error

        return _feedback_response(prediction)

    def get_status(self, prediction_id: int) -> PredictionFeedbackStatus:
        try:
            prediction = self._predictions.get(prediction_id)
        except SQLAlchemyError as error:
            logger.exception(
                "Could not retrieve feedback for prediction %s.",
                prediction_id,
            )
            raise FeedbackDatabaseError(
                "Ground-truth feedback could not be retrieved."
            ) from error
        if prediction is None:
            raise FeedbackNotFoundError(prediction_id)
        return _feedback_status(prediction)

    def get_summary(self) -> PredictionFeedbackSummary:
        try:
            counts = self._predictions.feedback_summary()
        except SQLAlchemyError as error:
            logger.exception("Could not retrieve the feedback summary.")
            raise FeedbackDatabaseError(
                "Ground-truth feedback summary could not be retrieved."
            ) from error

        incorrect = counts.feedback_received - counts.correct_predictions
        accuracy = (
            counts.correct_predictions / counts.feedback_received
            if counts.feedback_received
            else None
        )
        return PredictionFeedbackSummary(
            total_predictions=counts.total_predictions,
            feedback_received=counts.feedback_received,
            feedback_pending=(
                counts.total_predictions - counts.feedback_received
            ),
            correct_predictions=counts.correct_predictions,
            incorrect_predictions=incorrect,
            verified_accuracy=accuracy,
        )


def _feedback_status(prediction: Prediction) -> PredictionFeedbackStatus:
    actual_label = prediction.actual_label
    predicted_class = prediction.predicted_class
    feedback_received = prediction.feedback_received
    return PredictionFeedbackStatus(
        prediction_id=prediction.id,
        predicted_class=predicted_class,
        actual_label=actual_label,
        feedback_received=feedback_received,
        was_correct=(
            predicted_class == actual_label
            if feedback_received and actual_label is not None
            else None
        ),
        model_version=prediction.model_version,
    )


def _feedback_response(prediction: Prediction) -> PredictionFeedbackResponse:
    return PredictionFeedbackResponse(
        **_feedback_status(prediction).model_dump(),
        updated_at=prediction.updated_at,
    )
