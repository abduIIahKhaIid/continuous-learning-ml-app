from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.ml.config import ContinuousTrainingConfig
from app.ml.continuous_training import (
    ContinuousTrainingCoordinator,
    run_reserved_training,
)
from app.ml.model_loader import ModelLoader, model_loader
from app.repositories.data import SqlAlchemyDataRepository
from app.repositories.prediction import SqlAlchemyPredictionRepository
from app.repositories.training import SqlAlchemyTrainingRepository
from app.services.feedback import FeedbackService


async def get_data_repository(
    session: Annotated[Session, Depends(get_db)],
) -> SqlAlchemyDataRepository:
    return SqlAlchemyDataRepository(session)


DataRepositoryDependency = Annotated[
    SqlAlchemyDataRepository,
    Depends(get_data_repository),
]


async def get_prediction_repository(
    session: Annotated[Session, Depends(get_db)],
) -> SqlAlchemyPredictionRepository:
    return SqlAlchemyPredictionRepository(session)


async def get_training_repository(
    session: Annotated[Session, Depends(get_db)],
) -> SqlAlchemyTrainingRepository:
    return SqlAlchemyTrainingRepository(session)


async def get_model_loader() -> ModelLoader:
    return model_loader


async def get_model_directory() -> Path:
    return get_settings().resolved_model_dir


async def get_feedback_service(
    session: Annotated[Session, Depends(get_db)],
) -> FeedbackService:
    return FeedbackService(
        session=session,
        prediction_repository=SqlAlchemyPredictionRepository(session),
        data_repository=SqlAlchemyDataRepository(session),
    )


async def get_continuous_training_coordinator(
    session: Annotated[Session, Depends(get_db)],
) -> ContinuousTrainingCoordinator:
    return ContinuousTrainingCoordinator(
        SqlAlchemyTrainingRepository(session),
        ContinuousTrainingConfig.from_settings(get_settings()),
    )


async def get_continuous_training_config() -> ContinuousTrainingConfig:
    return ContinuousTrainingConfig.from_settings(get_settings())


async def get_training_job_runner() -> Callable[[int], None]:
    return run_reserved_training


PredictionRepositoryDependency = Annotated[
    SqlAlchemyPredictionRepository,
    Depends(get_prediction_repository),
]
TrainingRepositoryDependency = Annotated[
    SqlAlchemyTrainingRepository,
    Depends(get_training_repository),
]
ModelLoaderDependency = Annotated[ModelLoader, Depends(get_model_loader)]
ModelDirectoryDependency = Annotated[Path, Depends(get_model_directory)]
FeedbackServiceDependency = Annotated[
    FeedbackService, Depends(get_feedback_service)
]
ContinuousTrainingCoordinatorDependency = Annotated[
    ContinuousTrainingCoordinator,
    Depends(get_continuous_training_coordinator),
]
ContinuousTrainingConfigDependency = Annotated[
    ContinuousTrainingConfig,
    Depends(get_continuous_training_config),
]
TrainingJobRunnerDependency = Annotated[
    Callable[[int], None],
    Depends(get_training_job_runner),
]
