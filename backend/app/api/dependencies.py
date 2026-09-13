from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.core.config import get_settings
from app.ml.model_loader import ModelLoader, model_loader
from app.repositories.data import SqlAlchemyDataRepository
from app.repositories.prediction import SqlAlchemyPredictionRepository
from app.repositories.training import SqlAlchemyTrainingRepository


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
