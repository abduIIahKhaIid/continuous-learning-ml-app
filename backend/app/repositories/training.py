from typing import Any, Protocol

from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.sample import Sample, utc_now
from app.models.model_data_profile import ModelDataProfile
from app.models.model_event import ModelEvent
from app.models.training_run import TrainingRun

SERVABLE_STATUSES = ("completed", "promoted")
IN_PROGRESS_STATUSES = ("queued", "running", "training")
AUTOMATIC_CONCURRENCY_SLOT = "automatic"


class ModelVersionConflictError(Exception):
    pass


class AutomaticTrainingAlreadyReservedError(Exception):
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

    def list_verified_samples(self) -> list[Sample]:
        statement = (
            select(Sample)
            .where(
                Sample.source_prediction_id.is_not(None),
                Sample.label.in_((0, 1)),
            )
            .order_by(Sample.id)
        )
        return list(self._session.scalars(statement))

    def list_verified_samples_by_ids(
        self, sample_ids: list[int]
    ) -> list[Sample]:
        statement = (
            select(Sample)
            .where(
                Sample.id.in_(sample_ids),
                Sample.source_prediction_id.is_not(None),
                Sample.label.in_((0, 1)),
            )
            .order_by(Sample.id)
        )
        return list(self._session.scalars(statement))

    def list_unused_verified_samples(self) -> list[Sample]:
        statement = (
            select(Sample)
            .where(
                Sample.source_prediction_id.is_not(None),
                Sample.label.in_((0, 1)),
                Sample.used_for_training.is_(False),
                Sample.last_triggered_training_run_id.is_(None),
            )
            .order_by(Sample.id)
        )
        return list(self._session.scalars(statement))

    def count_unused_verified_samples(self) -> int:
        statement = select(func.count(Sample.id)).where(
            Sample.source_prediction_id.is_not(None),
            Sample.label.in_((0, 1)),
            Sample.used_for_training.is_(False),
            Sample.last_triggered_training_run_id.is_(None),
        )
        return int(self._session.scalar(statement) or 0)

    def list_model_versions(
        self, *, completed_only: bool = False
    ) -> list[str]:
        statement = select(TrainingRun.model_version)
        if completed_only:
            statement = statement.where(
                TrainingRun.status.in_(("completed", "promoted", "rejected"))
            )
        return list(self._session.scalars(statement))

    def get_model_for_inference(self) -> TrainingRun | None:
        active = self.get_active_model()
        if active is not None:
            return active
        statement = (
            select(TrainingRun)
            .where(TrainingRun.status.in_(SERVABLE_STATUSES))
            .order_by(TrainingRun.id.desc())
            .limit(1)
        )
        return self._session.scalar(statement)

    def get_active_model(self) -> TrainingRun | None:
        statement = (
            select(TrainingRun)
            .where(
                TrainingRun.status.in_(SERVABLE_STATUSES),
                TrainingRun.is_active.is_(True),
            )
            .order_by(TrainingRun.id.desc())
            .limit(1)
        )
        return self._session.scalar(statement)

    def get_by_model_version(self, model_version: str) -> TrainingRun | None:
        statement = select(TrainingRun).where(
            TrainingRun.model_version == model_version
        )
        return self._session.scalar(statement)

    def get_run(self, run_id: int) -> TrainingRun | None:
        return self._session.get(TrainingRun, run_id)

    def get_last_run(self) -> TrainingRun | None:
        return self._session.scalar(
            select(TrainingRun).order_by(TrainingRun.id.desc()).limit(1)
        )

    def get_current_run(self) -> TrainingRun | None:
        return self._session.scalar(
            select(TrainingRun)
            .where(TrainingRun.status.in_(IN_PROGRESS_STATUSES))
            .order_by(TrainingRun.id.desc())
            .limit(1)
        )

    def get_latest_terminal_run(self) -> TrainingRun | None:
        return self._session.scalar(
            select(TrainingRun)
            .where(
                TrainingRun.status.in_(
                    ("completed", "failed", "rejected", "promoted")
                )
            )
            .order_by(TrainingRun.id.desc())
            .limit(1)
        )

    def list_runs(self, *, skip: int, limit: int) -> list[TrainingRun]:
        statement = (
            select(TrainingRun)
            .order_by(TrainingRun.id.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def list_recoverable_queued_runs(self, *, limit: int = 100) -> list[TrainingRun]:
        statement = (
            select(TrainingRun)
            .where(
                TrainingRun.status == "queued",
                (
                    TrainingRun.celery_task_id.is_(None)
                    | TrainingRun.dispatch_error.is_not(None)
                ),
            )
            .order_by(TrainingRun.id)
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def count_runs_by_status(self, statuses: tuple[str, ...]) -> int:
        return int(
            self._session.scalar(
                select(func.count(TrainingRun.id)).where(
                    TrainingRun.status.in_(statuses)
                )
            )
            or 0
        )

    def record_dispatch(self, run_id: int, task_id: str) -> None:
        run = self._session.get(TrainingRun, run_id)
        if run is None:
            raise LookupError(f"Training run {run_id} no longer exists.")
        run.celery_task_id = task_id
        run.dispatched_at = utc_now()
        run.dispatch_error = None
        try:
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

    def record_dispatch_failure(self, run_id: int, message: str) -> None:
        self._session.rollback()
        run = self._session.get(TrainingRun, run_id)
        if run is None or run.status != "queued":
            return
        run.celery_task_id = None
        run.dispatch_error = message[:1000]
        self._session.commit()

    def update_progress(self, run_id: int, stage: str) -> None:
        run = self._session.get(TrainingRun, run_id)
        if run is None or run.status not in IN_PROGRESS_STATUSES:
            return
        run.progress_stage = stage
        run.heartbeat_at = utc_now()
        self._session.commit()

    def prepare_retry(
        self, run_id: int, message: str, *, requeue: bool = True
    ) -> None:
        run = self._session.get(TrainingRun, run_id)
        if run is None or run.status in ("completed", "failed", "rejected", "promoted"):
            return
        if requeue:
            run.status = "queued"
            run.progress_stage = "queued"
        run.retry_count += 1
        run.last_retry_at = utc_now()
        run.heartbeat_at = utc_now()
        run.error_message = message[:2000]
        self._session.commit()

    def is_run_stale(self, run: TrainingRun, *, cutoff: datetime) -> bool:
        heartbeat = run.heartbeat_at or run.started_at
        if run.status != "running" or heartbeat is None:
            return False
        if heartbeat.tzinfo is None and cutoff.tzinfo is not None:
            cutoff = cutoff.replace(tzinfo=None)
        return heartbeat < cutoff

    def has_training_in_progress(self) -> bool:
        statement = select(func.count(TrainingRun.id)).where(
            TrainingRun.status.in_(IN_PROGRESS_STATUSES)
        )
        return bool(self._session.scalar(statement))

    def rollback_read_transaction(self) -> None:
        self._session.rollback()

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
            trigger_type="manual",
            status="running",
        )
        self._session.add(run)
        try:
            self._session.commit()
        except IntegrityError as error:
            self._session.rollback()
            raise ModelVersionConflictError(model_version) from error
        return run

    def reserve_automatic_training_run(
        self,
        *,
        training_batch_id: str,
        model_version: str,
        trigger_sample_ids: list[int],
        training_sample_ids: list[int],
        total_training_sample_count: int,
        active_model_version: str | None,
        random_state: int,
        algorithm: str,
    ) -> TrainingRun:
        run = TrainingRun(
            training_batch_id=training_batch_id,
            model_version=model_version,
            algorithm=algorithm,
            training_sample_count=total_training_sample_count,
            random_state=random_state,
            trigger_type="automatic_threshold",
            trigger_new_sample_count=len(trigger_sample_ids),
            trigger_sample_ids=trigger_sample_ids,
            data_selection=training_sample_ids,
            active_model_version_before=active_model_version,
            concurrency_slot=AUTOMATIC_CONCURRENCY_SLOT,
            status="queued",
        )
        self._session.add(run)
        try:
            self._session.commit()
        except IntegrityError as error:
            self._session.rollback()
            raise AutomaticTrainingAlreadyReservedError from error
        return run

    def mark_run_running(self, run_id: int) -> TrainingRun | None:
        now = utc_now()
        result = self._session.execute(
            update(TrainingRun)
            .where(
                TrainingRun.id == run_id,
                TrainingRun.status == "queued",
            )
            .values(
                status="running",
                progress_stage="preparing_data",
                started_at=now,
                heartbeat_at=now,
                dispatch_error=None,
            )
        )
        self._session.commit()
        if result.rowcount != 1:
            return None
        return self._session.get(TrainingRun, run_id)

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
        reference_profiles: list[dict[str, Any]],
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
            if self.get_active_model() is None:
                run.is_active = True
                run.promoted_at = utc_now()
            run.error_message = None
            run.completed_at = utc_now()
            run.progress_stage = "completed"
            run.heartbeat_at = utc_now()
            self._add_profiles(run.model_version, reference_profiles)
            self._session.add(
                ModelEvent(
                    event_type="trained",
                    model_version=run.model_version,
                    reason="Manual baseline training completed.",
                )
            )
            if run.is_active:
                self._session.add(
                    ModelEvent(
                        event_type="promoted",
                        model_version=run.model_version,
                        reason="First available model activated.",
                    )
                )
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise
        self._session.refresh(run)
        return run

    def finalize_automatic_run(
        self,
        *,
        run_id: int,
        all_sample_ids: list[int],
        evaluation_sample_ids: list[int],
        model_path: str,
        artifact_checksum: str,
        artifact_format: str,
        parameters: dict[str, Any],
        train_sample_count: int,
        test_sample_count: int,
        metrics: dict[str, Any],
        active_metrics: dict[str, Any] | None,
        promoted: bool,
        rejection_reason: str | None,
        reference_profiles: list[dict[str, Any]],
    ) -> TrainingRun:
        run = self._session.get(TrainingRun, run_id)
        if run is None:
            raise LookupError(f"Training run {run_id} no longer exists.")
        if run.status != "running":
            raise RuntimeError(f"Training run {run_id} is not running.")

        now = utc_now()
        try:
            self._session.execute(
                update(Sample)
                .where(Sample.id.in_(all_sample_ids))
                .values(
                    used_for_training=True,
                    training_batch_id=run.training_batch_id,
                    model_version=run.model_version,
                )
            )
            self._session.execute(
                update(Sample)
                .where(Sample.id.in_(run.trigger_sample_ids or []))
                .values(last_triggered_training_run_id=run.id)
            )
            if promoted:
                self._session.execute(
                    update(TrainingRun)
                    .where(TrainingRun.is_active.is_(True))
                    .values(is_active=False)
                )
            run.model_path = model_path
            run.artifact_checksum = artifact_checksum
            run.artifact_format = artifact_format
            run.parameters = parameters
            run.data_selection = all_sample_ids
            run.evaluation_sample_ids = evaluation_sample_ids
            run.train_sample_count = train_sample_count
            run.test_sample_count = test_sample_count
            run.accuracy = metrics["accuracy"]
            run.precision = metrics["precision"]
            run.recall = metrics["recall"]
            run.f1_score = metrics["f1_score"]
            run.roc_auc = metrics["roc_auc"]
            run.confusion_matrix = metrics["confusion_matrix"]
            run.active_comparison_metrics = active_metrics
            run.status = "promoted" if promoted else "rejected"
            run.is_active = promoted
            run.promoted_at = now if promoted else None
            run.rejection_reason = rejection_reason
            run.error_message = None
            run.completed_at = now
            run.concurrency_slot = None
            run.progress_stage = "completed"
            run.heartbeat_at = now
            self._add_profiles(run.model_version, reference_profiles)
            self._session.add(
                ModelEvent(
                    event_type="trained",
                    model_version=run.model_version,
                    previous_model_version=run.active_model_version_before,
                    reason="Automatic candidate training completed.",
                )
            )
            self._session.add(
                ModelEvent(
                    event_type="promoted" if promoted else "rejected",
                    model_version=run.model_version,
                    previous_model_version=run.active_model_version_before,
                    reason=(
                        "Candidate passed the configured promotion gate."
                        if promoted
                        else (rejection_reason or "Promotion gate rejected candidate.")[:500]
                    ),
                )
            )
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
        run.is_active = False
        run.error_message = error_message[:2000]
        run.completed_at = utc_now()
        run.concurrency_slot = None
        run.progress_stage = "failed"
        run.heartbeat_at = utc_now()
        self._session.commit()

    def _add_profiles(
        self, model_version: str, profiles: list[dict[str, Any]]
    ) -> None:
        self._session.add_all(
            [
                ModelDataProfile(model_version=model_version, **profile)
                for profile in profiles
            ]
        )
