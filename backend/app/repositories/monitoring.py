from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.model_data_profile import ModelDataProfile
from app.models.monitoring_snapshot import MonitoringSnapshot
from app.models.prediction import Prediction


class SqlAlchemyMonitoringRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_profiles(self, model_version: str) -> list[ModelDataProfile]:
        return list(
            self._session.scalars(
                select(ModelDataProfile)
                .where(ModelDataProfile.model_version == model_version)
                .order_by(ModelDataProfile.feature_name)
            )
        )

    def list_recent_predictions(
        self, model_version: str, *, limit: int
    ) -> list[Prediction]:
        return list(
            self._session.scalars(
                select(Prediction)
                .where(Prediction.model_version == model_version)
                .order_by(Prediction.id.desc())
                .limit(limit)
            )
        )

    def list_recent_verified_predictions(
        self, model_version: str, *, limit: int
    ) -> list[Prediction]:
        return list(
            self._session.scalars(
                select(Prediction)
                .where(
                    Prediction.model_version == model_version,
                    Prediction.feedback_received.is_(True),
                    Prediction.actual_label.is_not(None),
                )
                .order_by(Prediction.id.desc())
                .limit(limit)
            )
        )

    def save_snapshots(
        self, snapshots: list[dict[str, Any]]
    ) -> list[MonitoringSnapshot]:
        rows = [MonitoringSnapshot(**snapshot) for snapshot in snapshots]
        self._session.add_all(rows)
        self._session.commit()
        for row in rows:
            self._session.refresh(row)
        return rows
