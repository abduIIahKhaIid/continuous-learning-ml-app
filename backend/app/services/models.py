import logging
from pathlib import Path

from app.ml.model_loader import ModelLoader
from app.repositories.models import SqlAlchemyModelRepository
from app.schemas.models import (
    DataProfileRead,
    ModelComparisonRead,
    ModelDetailRead,
    ModelEventRead,
    ModelSummaryRead,
    RollbackRead,
)

logger = logging.getLogger(__name__)


class ModelNotFoundError(LookupError):
    pass


class ModelRollbackError(RuntimeError):
    pass


class ModelService:
    def __init__(
        self,
        *,
        repository: SqlAlchemyModelRepository,
        loader: ModelLoader,
        model_dir: Path,
    ) -> None:
        self._repository = repository
        self._loader = loader
        self._model_dir = model_dir

    def list_models(self, *, skip: int, limit: int) -> list[ModelSummaryRead]:
        return [
            self._summary(run)
            for run in self._repository.list_models(skip=skip, limit=limit)
        ]

    def get_model(self, model_version: str) -> ModelDetailRead:
        run = self._require_model(model_version)
        profiles = self._repository.list_profiles(model_version)
        events = self._repository.list_events(model_version)
        return ModelDetailRead(
            **self._summary(run).model_dump(),
            parameters=run.parameters,
            confusion_matrix=run.confusion_matrix,
            data_profiles=[
                DataProfileRead.model_validate(profile, from_attributes=True)
                for profile in profiles
            ],
            events=[
                ModelEventRead.model_validate(event, from_attributes=True)
                for event in events
            ],
        )

    def compare(self, model_a: str, model_b: str) -> ModelComparisonRead:
        first = self._require_model(model_a)
        second = self._require_model(model_b)
        fields = ("accuracy", "precision", "recall", "f1_score", "roc_auc")
        differences = {
            field: (
                getattr(first, field) - getattr(second, field)
                if getattr(first, field) is not None
                and getattr(second, field) is not None
                else None
            )
            for field in fields
        }
        return ModelComparisonRead(
            model_a=self._summary(first),
            model_b=self._summary(second),
            metric_differences=differences,
            comparison_notice=(
                "Historical validation metrics may use different evaluation "
                "datasets and are not necessarily statistically comparable."
            ),
        )

    def rollback(self, model_version: str, reason: str | None) -> RollbackRead:
        logger.info("Model rollback requested: target=%s", model_version)
        target = self._require_model(model_version)
        if target.status not in ("completed", "promoted"):
            raise ModelRollbackError(
                "Only completed or promoted model versions can be activated."
            )
        active = self._repository.get_active_model()
        if active is not None and active.model_version == model_version:
            raise ModelRollbackError("The requested model is already active.")
        self._loader.validate_registered_run(
            target, model_dir=self._model_dir, expected_feature_count=3
        )
        event = self._repository.activate_rollback(
            target=target, previous=active, reason=reason
        )
        self._loader.invalidate_cache()
        logger.warning(
            "Active model changed by rollback: previous=%s target=%s",
            active.model_version if active else None,
            target.model_version,
        )
        return RollbackRead(
            active_model_version=target.model_version,
            previous_model_version=(active.model_version if active else None),
            event_id=event.id,
            message=f"{target.model_version} is now the active model.",
        )

    def _require_model(self, model_version: str):
        run = self._repository.get_model(model_version)
        if run is None:
            raise ModelNotFoundError(model_version)
        return run

    @staticmethod
    def _summary(run) -> ModelSummaryRead:
        return ModelSummaryRead.model_validate(run, from_attributes=True)
