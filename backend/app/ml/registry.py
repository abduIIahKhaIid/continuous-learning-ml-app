import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import joblib
from sklearn.pipeline import Pipeline

from app.models.training_run import TrainingRun
from app.repositories.training import (
    ModelVersionConflictError,
    TrainingRepository,
)

MODEL_VERSION_PATTERN = re.compile(r"model_v(\d+)")


class ModelRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactInfo:
    path: Path
    checksum: str
    format: str = "joblib"


def calculate_artifact_checksum(artifact_path: Path) -> str:
    checksum_builder = hashlib.sha256()
    with artifact_path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            checksum_builder.update(chunk)
    return checksum_builder.hexdigest()


def generate_next_model_version(existing_versions: list[str]) -> str:
    version_numbers = [
        int(match.group(1))
        for version in existing_versions
        if (match := MODEL_VERSION_PATTERN.fullmatch(version)) is not None
    ]
    return f"model_v{max(version_numbers, default=0) + 1}"


def reserve_training_run(
    repository: TrainingRepository,
    *,
    algorithm: str,
    training_sample_count: int,
    random_state: int,
) -> TrainingRun:
    training_batch_id = f"batch_{uuid4().hex}"
    for _ in range(10):
        model_version = generate_next_model_version(
            repository.list_model_versions()
        )
        try:
            return repository.create_training_run(
                training_batch_id=training_batch_id,
                model_version=model_version,
                algorithm=algorithm,
                training_sample_count=training_sample_count,
                random_state=random_state,
            )
        except ModelVersionConflictError:
            continue
    raise ModelRegistryError(
        "Could not reserve a unique model version after 10 attempts."
    )


def save_trained_model(
    pipeline: Pipeline,
    *,
    model_version: str,
    model_dir: Path,
) -> ArtifactInfo:
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / f"{model_version}.joblib"
    try:
        artifact_file = artifact_path.open("xb")
    except FileExistsError:
        raise

    try:
        with artifact_file:
            joblib.dump(pipeline, artifact_file)
    except Exception:
        artifact_path.unlink(missing_ok=True)
        raise

    checksum = calculate_artifact_checksum(artifact_path)
    return ArtifactInfo(path=artifact_path, checksum=checksum)


def list_available_model_versions(
    repository: TrainingRepository,
) -> list[str]:
    versions = repository.list_model_versions(completed_only=True)
    parsed_versions = [
        (int(match.group(1)), version)
        for version in versions
        if (match := MODEL_VERSION_PATTERN.fullmatch(version)) is not None
    ]
    return [version for _, version in sorted(parsed_versions)]
