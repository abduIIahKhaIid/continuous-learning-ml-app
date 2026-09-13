from typing import Any, Protocol

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.sample import Sample
from app.models.training_run import TrainingRun


class ModelVersionConflictError(Exception):
    pass


class TrainingRepository(Protocol):
    def list_labelled_samples(self) -> list[Sample]: ...

    def list_model_versions(
        self, *, completed_only: bool = False
    ) -> list[str]: ...

    def get_model_for_inference(self) -> TrainingRun | None: ...

    def create_training_run(
        self,
        *,
        training_batch_id: str,
        model_version: str,
        algorithm: str,
        training_sample_count: int,
        random_state: int,
    ) -> TrainingRun: ...

    def complete_training_run(
        self,
        *,
        run_id: int,
        sample_ids: list[int],
        model_path: str,
        artifact_checksum: str,
        artifact_format: str,
        parameters: dict[str, Any],
        train_sample_count: int,
        test_sample_count: int,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        roc_auc: float | None,
        confusion_matrix: list[list[int]],
    ) -> TrainingRun: ...

    def fail_training_run(self, run_id: int, error_message: str) -> None: ...


class SqlAlchemyTrainingRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def list_labelled_samples(self) -> list[Sample]:
        statement = (
            select(Sample)
            .where(Sample.label.is_not(None))
            .order_by(Sample.id)
        )
        return list(self._session.scalars(statement))

    def list_model_versions(
        self, *, completed_only: bool = False
    ) -> list[str]:
        statement = select(TrainingRun.model_version)
        if completed_only:
            statement = statement.where(TrainingRun.status == "completed")
        return list(self._session.scalars(statement))

    def get_model_for_inference(self) -> TrainingRun | None:
        active_statement = (
            select(TrainingRun)
            .where(
                TrainingRun.status == "completed",
                TrainingRun.is_active.is_(True),
            )
            .order_by(TrainingRun.id.desc())
            .limit(1)
        )
        active_run = self._session.scalar(active_statement)
        if active_run is not None:
            return active_run

        fallback_statement = (
            select(TrainingRun)
            .where(TrainingRun.status == "completed")
            .order_by(TrainingRun.id.desc())
            .limit(1)
        )
        return self._session.scalar(fallback_statement)

    def create_training_run(
        self,
        *,
        training_batch_id: str,
        model_version: str,
        algorithm: str,
        training_sample_count: int,
        random_state: int,
    ) -> TrainingRun:
        run = TrainingRun(
            training_batch_id=training_batch_id,
            model_version=model_version,
            algorithm=algorithm,
            training_sample_count=training_sample_count,
            random_state=random_state,
            status="training",
        )
        self._session.add(run)
        try:
            self._session.commit()
        except IntegrityError as error:
            self._session.rollback()
            raise ModelVersionConflictError(model_version) from error
        self._session.refresh(run)
        return run

    def complete_training_run(
        self,
        *,
        run_id: int,
        sample_ids: list[int],
        model_path: str,
        artifact_checksum: str,
        artifact_format: str,
        parameters: dict[str, Any],
        train_sample_count: int,
        test_sample_count: int,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        roc_auc: float | None,
        confusion_matrix: list[list[int]],
    ) -> TrainingRun:
        run = self._session.get(TrainingRun, run_id)
        if run is None:
            raise LookupError(f"Training run {run_id} no longer exists.")

        try:
            self._session.execute(
                update(Sample)
                .where(Sample.id.in_(sample_ids))
                .values(
                    used_for_training=True,
                    training_batch_id=run.training_batch_id,
                    model_version=run.model_version,
                )
            )
            run.model_path = model_path
            run.artifact_checksum = artifact_checksum
            run.artifact_format = artifact_format
            run.parameters = parameters
            run.data_selection = sample_ids
            run.train_sample_count = train_sample_count
            run.test_sample_count = test_sample_count
            run.accuracy = accuracy
            run.precision = precision
            run.recall = recall
            run.f1_score = f1_score
            run.roc_auc = roc_auc
            run.confusion_matrix = confusion_matrix
            run.status = "completed"
            run.error_message = None
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise
        self._session.refresh(run)
        return run

    def fail_training_run(self, run_id: int, error_message: str) -> None:
        self._session.rollback()
        run = self._session.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = "failed"
        run.error_message = error_message[:2000]
        self._session.commit()
