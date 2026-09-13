import logging
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.ml.model_loader import ModelLoader, ModelUnavailableError
from app.monitoring.drift import calculate_psi, classify_psi, worst_status
from app.monitoring.performance import (
    calculate_verified_metrics,
    classify_performance_drop,
)
from app.repositories.monitoring import SqlAlchemyMonitoringRepository
from app.repositories.training import SqlAlchemyTrainingRepository
from app.schemas.monitoring import (
    DriftRead,
    FeatureDriftRead,
    MonitoringCheckRead,
    MonitoringHealthRead,
    MonitoringSnapshotRead,
    PerformanceRead,
)

logger = logging.getLogger(__name__)


class MonitoringService:
    def __init__(
        self,
        *,
        monitoring_repository: SqlAlchemyMonitoringRepository,
        training_repository: SqlAlchemyTrainingRepository,
        loader: ModelLoader,
        settings: Settings,
        model_dir: Path,
    ) -> None:
        self._monitoring = monitoring_repository
        self._training = training_repository
        self._loader = loader
        self._settings = settings
        self._model_dir = model_dir

    def data_drift(self) -> DriftRead:
        active = self._training.get_active_model()
        if active is None:
            logger.info("Drift check has no active model reference profile.")
            return DriftRead(
                model_version=None,
                status="insufficient_data",
                sample_count=0,
                minimum_samples=self._settings.drift_min_samples,
                window_size=self._settings.drift_window_size,
                features=[],
                samples_analyzed=0,
                overall_status="insufficient_data",
            )
        profiles = self._monitoring.list_profiles(active.model_version)
        predictions = self._monitoring.list_recent_predictions(
            active.model_version, limit=self._settings.drift_window_size
        )
        sample_count = len(predictions)
        values_by_feature = {
            "feature_1": [row.feature_1 for row in predictions],
            "feature_2": [row.feature_2 for row in predictions],
            "feature_3": [row.feature_3 for row in predictions],
        }
        features: list[FeatureDriftRead] = []
        for profile in profiles:
            if sample_count < self._settings.drift_min_samples:
                psi, status = None, "insufficient_data"
            else:
                psi = calculate_psi(
                    profile, values_by_feature.get(profile.feature_name, [])
                )
                status = classify_psi(
                    psi,
                    warning_threshold=self._settings.drift_psi_warning,
                    critical_threshold=self._settings.drift_psi_critical,
                )
            features.append(
                FeatureDriftRead(
                    feature_name=profile.feature_name,
                    psi=psi,
                    status=status,
                    reference_samples=profile.sample_count,
                    current_samples=sample_count,
                )
            )
        overall = worst_status([feature.status for feature in features])
        log = logger.warning if overall in ("warning", "critical") else logger.info
        log(
            "Data drift calculated: model=%s status=%s samples=%s",
            active.model_version,
            overall,
            sample_count,
        )
        return DriftRead(
            model_version=active.model_version,
            status=overall,
            sample_count=sample_count,
            minimum_samples=self._settings.drift_min_samples,
            window_size=self._settings.drift_window_size,
            features=features,
            samples_analyzed=sample_count,
            overall_status=overall,
        )

    def performance(self) -> PerformanceRead:
        active = self._training.get_active_model()
        metric_name = self._settings.primary_promotion_metric
        if active is None:
            return self._empty_performance(metric_name)
        predictions = self._monitoring.list_recent_predictions(
            active.model_version,
            limit=self._settings.performance_window_size,
        )
        coverage_verified = [
            row
            for row in predictions
            if row.feedback_received and row.actual_label is not None
        ]
        verified = self._monitoring.list_recent_verified_predictions(
            active.model_version,
            limit=self._settings.performance_window_size,
        )
        coverage = (
            len(coverage_verified) / len(predictions) if predictions else 0.0
        )
        baseline = getattr(active, metric_name, None)
        if (
            len(verified) < self._settings.performance_min_feedback_samples
            or baseline is None
        ):
            logger.info(
                "Performance monitoring has insufficient verified feedback: "
                "model=%s verified=%s minimum=%s",
                active.model_version,
                len(verified),
                self._settings.performance_min_feedback_samples,
            )
            return PerformanceRead(
                model_version=active.model_version,
                status="insufficient_data",
                total_predictions=len(predictions),
                verified_samples=len(verified),
                minimum_samples=(
                    self._settings.performance_min_feedback_samples
                ),
                window_size=self._settings.performance_window_size,
                feedback_coverage=coverage,
                primary_metric=metric_name,
                baseline_value=baseline,
                current_value=None,
                metric_drop=None,
                metrics=None,
                samples_with_feedback=len(verified),
                baseline_metric=baseline,
                current_metric=None,
            )
        metrics = calculate_verified_metrics(verified)
        current = metrics.get(metric_name)
        if current is None:
            status, drop = "insufficient_data", None
        else:
            status, drop = classify_performance_drop(
                baseline,
                current,
                warning_drop=self._settings.performance_warning_drop,
                critical_drop=self._settings.performance_critical_drop,
            )
        log = logger.warning if status in ("warning", "critical") else logger.info
        log(
            "Performance monitoring calculated: model=%s status=%s verified=%s",
            active.model_version,
            status,
            len(verified),
        )
        return PerformanceRead(
            model_version=active.model_version,
            status=status,
            total_predictions=len(predictions),
            verified_samples=len(verified),
            minimum_samples=self._settings.performance_min_feedback_samples,
            window_size=self._settings.performance_window_size,
            feedback_coverage=coverage,
            primary_metric=metric_name,
            baseline_value=baseline,
            current_value=current,
            metric_drop=drop,
            metrics=metrics,
            samples_with_feedback=len(verified),
            baseline_metric=baseline,
            current_metric=current,
        )

    def health(
        self,
        *,
        drift: DriftRead | None = None,
        performance: PerformanceRead | None = None,
    ) -> MonitoringHealthRead:
        active = self._training.get_active_model()
        drift = drift or self.data_drift()
        performance = performance or self.performance()
        artifact_available = False
        if active is not None:
            try:
                self._loader.validate_registered_run(
                    active, model_dir=self._model_dir
                )
                artifact_available = True
            except ModelUnavailableError:
                pass
        training_in_progress = self._training.has_training_in_progress()
        last_run = self._training.get_last_run()
        if active is None or not artifact_available:
            status, message = "unhealthy", "Active model artifact is unavailable."
        elif performance.status == "critical":
            status, message = "unhealthy", "Verified model performance is critical."
        elif drift.status in ("warning", "critical") or performance.status == "warning":
            status, message = "warning", "Monitoring thresholds need attention."
        else:
            status, message = "healthy", "Active model is available."
        return MonitoringHealthRead(
            status=status,
            model_status=status,
            model_version=active.model_version if active else None,
            artifact_available=artifact_available,
            data_drift_status=drift.status,
            performance_status=performance.status,
            feedback_coverage=performance.feedback_coverage,
            training_in_progress=training_in_progress,
            last_training_status=last_run.status if last_run else None,
            new_verified_samples=self._training.count_unused_verified_samples(),
            retrain_threshold=self._settings.retrain_min_new_samples,
            rollback_recommended=performance.status == "critical",
            message=message,
        )

    def check(self) -> MonitoringCheckRead:
        drift = self.data_drift()
        performance = self.performance()
        health = self.health(drift=drift, performance=performance)
        version = health.model_version
        payloads: list[tuple[str, str, int, dict[str, Any]]] = [
            ("data_drift", drift.status, drift.sample_count, drift.model_dump()),
            (
                "performance",
                performance.status,
                performance.verified_samples,
                performance.model_dump(),
            ),
            ("health", health.status, performance.total_predictions, health.model_dump()),
        ]
        rows = self._monitoring.save_snapshots(
            [
                {
                    "model_version": version,
                    "snapshot_type": kind,
                    "overall_status": status,
                    "sample_count": count,
                    "metrics_json": metrics,
                }
                for kind, status, count, metrics in payloads
            ]
        )
        return MonitoringCheckRead(
            drift=drift,
            performance=performance,
            health=health,
            snapshots=[
                MonitoringSnapshotRead(
                    id=row.id,
                    snapshot_type=row.snapshot_type,
                    status=row.overall_status,
                    created_at=row.created_at,
                )
                for row in rows
            ],
        )

    def _empty_performance(self, metric_name: str) -> PerformanceRead:
        return PerformanceRead(
            model_version=None,
            status="insufficient_data",
            total_predictions=0,
            verified_samples=0,
            minimum_samples=self._settings.performance_min_feedback_samples,
            window_size=self._settings.performance_window_size,
            feedback_coverage=0.0,
            primary_metric=metric_name,
            baseline_value=None,
            current_value=None,
            metric_drop=None,
            metrics=None,
            samples_with_feedback=0,
            baseline_metric=None,
            current_metric=None,
        )
