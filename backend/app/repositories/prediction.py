from typing import Protocol

from sqlalchemy import select
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
