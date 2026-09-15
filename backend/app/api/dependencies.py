from pathlib import Path
from typing import Annotated

from fastapi import Depends
from redis import Redis
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import get_db
from app.ml.config import ContinuousTrainingConfig
from app.ml.continuous_training import ContinuousTrainingCoordinator
from app.ml.model_loader import ModelLoader, model_loader
from app.repositories.data import SqlAlchemyDataRepository
from app.repositories.models import SqlAlchemyModelRepository
from app.repositories.monitoring import SqlAlchemyMonitoringRepository
from app.repositories.prediction import SqlAlchemyPredictionRepository
from app.repositories.training import SqlAlchemyTrainingRepository
from app.services.feedback import FeedbackService
from app.services.models import ModelService
from app.services.monitoring import MonitoringService
from app.services.infrastructure_health import InfrastructureHealthService
from app.services.training_dispatch import TrainingDispatcher
from app.workers.celery_app import celery_app
from app.workers.tasks import run_retraining_task


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


async def get_training_dispatcher(
    session: Annotated[Session, Depends(get_db)],
) -> TrainingDispatcher:
    settings = get_settings()
    return TrainingDispatcher(
        repository=SqlAlchemyTrainingRepository(session),
        task_sender=run_retraining_task,
        queue_name=settings.training_queue_name,
    )


async def get_infrastructure_health_service() -> InfrastructureHealthService:
    settings = get_settings()
    return InfrastructureHealthService(
        redis_client=Redis.from_url(settings.redis_url),
        celery_app=celery_app,
        timeout_seconds=settings.worker_health_timeout_seconds,
    )


async def get_monitoring_service(
    session: Annotated[Session, Depends(get_db)],
    loader: Annotated[ModelLoader, Depends(get_model_loader)],
) -> MonitoringService:
    settings = get_settings()
    return MonitoringService(
        monitoring_repository=SqlAlchemyMonitoringRepository(session),
        training_repository=SqlAlchemyTrainingRepository(session),
        loader=loader,
        settings=settings,
        model_dir=settings.resolved_model_dir,
    )


async def get_model_service(
    session: Annotated[Session, Depends(get_db)],
    loader: Annotated[ModelLoader, Depends(get_model_loader)],
) -> ModelService:
    settings = get_settings()
    return ModelService(
        repository=SqlAlchemyModelRepository(session),
        loader=loader,
        model_dir=settings.resolved_model_dir,
    )


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
TrainingDispatcherDependency = Annotated[
    TrainingDispatcher, Depends(get_training_dispatcher)
]
InfrastructureHealthDependency = Annotated[
    InfrastructureHealthService,
    Depends(get_infrastructure_health_service),
]
MonitoringServiceDependency = Annotated[
    MonitoringService, Depends(get_monitoring_service)
]
ModelServiceDependency = Annotated[ModelService, Depends(get_model_service)]
