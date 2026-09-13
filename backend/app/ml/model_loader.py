import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import RLock

import joblib
from sklearn.pipeline import Pipeline

from app.ml.registry import calculate_artifact_checksum
from app.repositories.training import TrainingRepository

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class NoModelAvailableError(ModelUnavailableError):
    pass


class ModelArtifactUnavailableError(ModelUnavailableError):
    def __init__(self, model_version: str) -> None:
        self.model_version = model_version
        super().__init__(f"Artifact unavailable for {model_version}.")


class ModelLoadError(ModelUnavailableError):
    def __init__(self, model_version: str) -> None:
        self.model_version = model_version
        super().__init__(f"Could not load {model_version}.")


@dataclass(frozen=True)
class ModelDescriptor:
    model_version: str
    model_path: str
    artifact_checksum: str
    algorithm: str
    created_at: datetime
    accuracy: float | None
    f1_score: float | None
    roc_auc: float | None


@dataclass(frozen=True)
class LoadedModel:
    pipeline: Pipeline
    descriptor: ModelDescriptor


class ModelLoader:
    def __init__(self) -> None:
        self._cached_model: LoadedModel | None = None
        self._lock = RLock()

    @property
    def loaded_model_version(self) -> str | None:
        if self._cached_model is None:
            return None
        return self._cached_model.descriptor.model_version

    def clear_cache(self) -> None:
        with self._lock:
            self._cached_model = None

    def load(
        self,
        repository: TrainingRepository,
        *,
        model_dir: Path,
    ) -> LoadedModel:
        run = repository.get_model_for_inference()
        if run is None:
            raise NoModelAvailableError("No completed model is available.")
        if (
            not run.model_path
            or not run.artifact_checksum
            or run.artifact_format != "joblib"
        ):
            logger.error(
                "Completed model %s has incomplete artifact metadata.",
                run.model_version,
            )
            raise ModelArtifactUnavailableError(run.model_version)

        descriptor = ModelDescriptor(
            model_version=run.model_version,
            model_path=run.model_path,
            artifact_checksum=run.artifact_checksum,
            algorithm=run.algorithm,
            created_at=run.created_at,
            accuracy=run.accuracy,
            f1_score=run.f1_score,
            roc_auc=run.roc_auc,
        )
        with self._lock:
            cached = self._cached_model
            if (
                cached is not None
                and cached.descriptor.model_version
                == descriptor.model_version
                and cached.descriptor.artifact_checksum
                == descriptor.artifact_checksum
            ):
                return cached

            artifact_path = Path(descriptor.model_path).resolve()
            allowed_directory = model_dir.resolve()
            try:
                artifact_path.relative_to(allowed_directory)
            except ValueError:
                logger.error(
                    "Model %s references an artifact outside MODEL_DIR.",
                    descriptor.model_version,
                )
                raise ModelArtifactUnavailableError(
                    descriptor.model_version
                ) from None

            if artifact_path.suffix != ".joblib":
                logger.error(
                    "Model %s references an unexpected artifact format.",
                    descriptor.model_version,
                )
                raise ModelArtifactUnavailableError(descriptor.model_version)

            if not artifact_path.is_file():
                logger.error(
                    "Artifact for model %s does not exist.",
                    descriptor.model_version,
                )
                raise ModelArtifactUnavailableError(descriptor.model_version)

            try:
                checksum = calculate_artifact_checksum(artifact_path)
            except OSError as error:
                logger.exception(
                    "Artifact for model %s could not be read.",
                    descriptor.model_version,
                )
                raise ModelArtifactUnavailableError(
                    descriptor.model_version
                ) from error
            if checksum != descriptor.artifact_checksum:
                logger.error(
                    "Artifact checksum mismatch for model %s.",
                    descriptor.model_version,
                )
                raise ModelLoadError(descriptor.model_version)

            try:
                pipeline = joblib.load(artifact_path)
            except Exception as error:
                logger.exception(
                    "Unable to deserialize model %s.",
                    descriptor.model_version,
                )
                raise ModelLoadError(descriptor.model_version) from error
            if not isinstance(pipeline, Pipeline):
                logger.error(
                    "Artifact for model %s is not an sklearn Pipeline.",
                    descriptor.model_version,
                )
                raise ModelLoadError(descriptor.model_version)

            loaded = LoadedModel(pipeline=pipeline, descriptor=descriptor)
            self._cached_model = loaded
            return loaded


model_loader = ModelLoader()
