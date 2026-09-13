from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.sample import Sample


class DataRepository(Protocol):
    def create(
        self,
        *,
        feature_1: float,
        feature_2: float,
        feature_3: float,
        label: int | None,
    ) -> Sample: ...

    def list(
        self,
        *,
        skip: int,
        limit: int,
        labelled_only: bool = False,
        unused_only: bool = False,
    ) -> list[Sample]: ...

    def get(self, record_id: int) -> Sample | None: ...

    def get_by_source_prediction_id(
        self, prediction_id: int
    ) -> Sample | None: ...

    def create_verified(
        self,
        *,
        feature_1: float,
        feature_2: float,
        feature_3: float,
        actual_label: int,
        source_prediction_id: int,
    ) -> Sample: ...


class SqlAlchemyDataRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        feature_1: float,
        feature_2: float,
        feature_3: float,
        label: int | None,
    ) -> Sample:
        sample = Sample(
            feature_1=feature_1,
            feature_2=feature_2,
            feature_3=feature_3,
            label=label,
        )
        self._session.add(sample)
        try:
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise
        self._session.refresh(sample)
        return sample

    def list(
        self,
        *,
        skip: int,
        limit: int,
        labelled_only: bool = False,
        unused_only: bool = False,
    ) -> list[Sample]:
        statement = select(Sample)
        if labelled_only:
            statement = statement.where(Sample.label.is_not(None))
        if unused_only:
            statement = statement.where(Sample.used_for_training.is_(False))
        statement = statement.order_by(Sample.id).offset(skip).limit(limit)
        return list(self._session.scalars(statement))

    def get(self, record_id: int) -> Sample | None:
        return self._session.get(Sample, record_id)

    def get_by_source_prediction_id(
        self, prediction_id: int
    ) -> Sample | None:
        statement = select(Sample).where(
            Sample.source_prediction_id == prediction_id
        )
        return self._session.scalar(statement)

    def create_verified(
        self,
        *,
        feature_1: float,
        feature_2: float,
        feature_3: float,
        actual_label: int,
        source_prediction_id: int,
    ) -> Sample:
        sample = Sample(
            feature_1=feature_1,
            feature_2=feature_2,
            feature_3=feature_3,
            label=actual_label,
            source_prediction_id=source_prediction_id,
            used_for_training=False,
            training_batch_id=None,
            model_version=None,
        )
        self._session.add(sample)
        self._session.flush()
        return sample
