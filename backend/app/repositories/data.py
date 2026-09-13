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

    def list(self, *, skip: int, limit: int) -> list[Sample]: ...

    def get(self, record_id: int) -> Sample | None: ...


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

    def list(self, *, skip: int, limit: int) -> list[Sample]:
        statement = (
            select(Sample)
            .order_by(Sample.id)
            .offset(skip)
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def get(self, record_id: int) -> Sample | None:
        return self._session.get(Sample, record_id)
