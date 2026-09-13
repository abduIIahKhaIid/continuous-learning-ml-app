from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.models.prediction import Prediction


class PredictionRepository(Protocol):
    def create(
        self,
        *,
        feature_1: float,
        feature_2: float,
        feature_3: float,
        predicted_class: int,
        prediction_probability: float | None,
        model_version: str,
    ) -> Prediction: ...

    def list(self, *, skip: int, limit: int) -> list[Prediction]: ...

    def get(self, prediction_id: int) -> Prediction | None: ...

    def get_for_feedback(self, prediction_id: int) -> Prediction | None: ...

    def set_feedback(
        self, prediction: Prediction, *, actual_label: int
    ) -> None: ...

    def feedback_summary(self) -> "FeedbackCounts": ...


@dataclass(frozen=True)
class FeedbackCounts:
    total_predictions: int
    feedback_received: int
    correct_predictions: int


class SqlAlchemyPredictionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        feature_1: float,
        feature_2: float,
        feature_3: float,
        predicted_class: int,
        prediction_probability: float | None,
        model_version: str,
    ) -> Prediction:
        prediction = Prediction(
            feature_1=feature_1,
            feature_2=feature_2,
            feature_3=feature_3,
            predicted_class=predicted_class,
            prediction_probability=prediction_probability,
            model_version=model_version,
            actual_label=None,
            feedback_received=False,
        )
        self._session.add(prediction)
        try:
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise
        self._session.refresh(prediction)
        return prediction

    def list(self, *, skip: int, limit: int) -> list[Prediction]:
        statement = (
            select(Prediction)
            .order_by(Prediction.id.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def get(self, prediction_id: int) -> Prediction | None:
        return self._session.get(Prediction, prediction_id)

    def get_for_feedback(self, prediction_id: int) -> Prediction | None:
        statement = (
            select(Prediction)
            .where(Prediction.id == prediction_id)
            .with_for_update()
        )
        return self._session.scalar(statement)

    def set_feedback(
        self, prediction: Prediction, *, actual_label: int
    ) -> None:
        prediction.actual_label = actual_label
        prediction.feedback_received = True

    def feedback_summary(self) -> FeedbackCounts:
        verified = (
            Prediction.feedback_received.is_(True)
            & Prediction.actual_label.is_not(None)
        )
        statement = select(
            func.count(Prediction.id),
            func.sum(case((verified, 1), else_=0)),
            func.sum(
                case(
                    (
                        verified
                        & (Prediction.predicted_class == Prediction.actual_label),
                        1,
                    ),
                    else_=0,
                )
            ),
        )
        total, received, correct = self._session.execute(statement).one()
        return FeedbackCounts(
            total_predictions=int(total or 0),
            feedback_received=int(received or 0),
            correct_predictions=int(correct or 0),
        )
