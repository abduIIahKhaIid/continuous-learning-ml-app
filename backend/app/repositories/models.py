from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models.model_data_profile import ModelDataProfile
from app.models.model_event import ModelEvent
from app.models.training_run import TrainingRun


class SqlAlchemyModelRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_models(self, *, skip: int, limit: int) -> list[TrainingRun]:
        return list(
            self._session.scalars(
                select(TrainingRun)
                .order_by(TrainingRun.id.desc())
                .offset(skip)
                .limit(limit)
            )
        )

    def get_model(self, model_version: str) -> TrainingRun | None:
        return self._session.scalar(
            select(TrainingRun).where(
                TrainingRun.model_version == model_version
            )
        )

    def get_active_model(self) -> TrainingRun | None:
        return self._session.scalar(
            select(TrainingRun)
            .where(TrainingRun.is_active.is_(True))
            .order_by(TrainingRun.id.desc())
            .limit(1)
        )

    def list_profiles(self, model_version: str) -> list[ModelDataProfile]:
        return list(
            self._session.scalars(
                select(ModelDataProfile)
                .where(ModelDataProfile.model_version == model_version)
                .order_by(ModelDataProfile.feature_name)
            )
        )

    def list_events(self, model_version: str) -> list[ModelEvent]:
        return list(
            self._session.scalars(
                select(ModelEvent)
                .where(ModelEvent.model_version == model_version)
                .order_by(ModelEvent.id.desc())
            )
        )

    def activate_rollback(
        self,
        *,
        target: TrainingRun,
        previous: TrainingRun | None,
        reason: str | None,
    ) -> ModelEvent:
        try:
            self._session.execute(
                update(TrainingRun)
                .where(TrainingRun.is_active.is_(True))
                .values(is_active=False)
            )
            target.is_active = True
            target.promoted_at = target.promoted_at or target.completed_at
            event = ModelEvent(
                event_type="rollback",
                model_version=target.model_version,
                previous_model_version=(
                    previous.model_version if previous else None
                ),
                reason=reason,
            )
            self._session.add(event)
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise
        self._session.refresh(target)
        self._session.refresh(event)
        return event

