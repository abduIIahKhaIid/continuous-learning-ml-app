from fastapi import APIRouter, HTTPException, Path, Query, status

from app.api.dependencies import DataRepositoryDependency
from app.schemas.data import DataCreate, DataRead, TrainingDataRead
from app.services.data import (
    DataNotFoundError,
    create_data,
    get_data,
    list_data,
    list_training_data,
)

router = APIRouter(prefix="/api", tags=["data"])


@router.post(
    "/data",
    response_model=DataRead,
    status_code=status.HTTP_201_CREATED,
)
async def submit_data(
    payload: DataCreate,
    repository: DataRepositoryDependency,
) -> DataRead:
    return create_data(payload, repository)


@router.get("/data", response_model=list[DataRead])
async def retrieve_data(
    repository: DataRepositoryDependency,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[DataRead]:
    return list_data(skip=skip, limit=limit, repository=repository)


@router.get("/data/{record_id}", response_model=DataRead)
async def retrieve_data_by_id(
    repository: DataRepositoryDependency,
    record_id: int = Path(gt=0),
) -> DataRead:
    try:
        return get_data(record_id, repository)
    except DataNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data record not found.",
        ) from error


@router.get("/training-data", response_model=list[TrainingDataRead])
async def retrieve_training_data(
    repository: DataRepositoryDependency,
    labelled_only: bool = Query(default=False),
    unused_only: bool = Query(default=False),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[TrainingDataRead]:
    return list_training_data(
        skip=skip,
        limit=limit,
        labelled_only=labelled_only,
        unused_only=unused_only,
        repository=repository,
    )
