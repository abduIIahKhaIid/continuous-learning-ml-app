import logging

from fastapi import APIRouter, HTTPException, Query, status

from app.api.dependencies import ModelServiceDependency
from app.ml.model_loader import ModelUnavailableError
from app.schemas.models import (
    ModelComparisonRead,
    ModelDetailRead,
    ModelSummaryRead,
    RollbackRead,
    RollbackRequest,
)
from app.services.models import ModelNotFoundError, ModelRollbackError

router = APIRouter(prefix="/api/models", tags=["model registry"])
logger = logging.getLogger(__name__)


@router.get("", response_model=list[ModelSummaryRead])
async def model_history(
    service: ModelServiceDependency,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[ModelSummaryRead]:
    return service.list_models(skip=skip, limit=limit)


@router.get("/compare", response_model=ModelComparisonRead)
async def compare_models(
    service: ModelServiceDependency,
    model_a: str = Query(min_length=1),
    model_b: str = Query(min_length=1),
) -> ModelComparisonRead:
    try:
        return service.compare(model_a, model_b)
    except ModelNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {error.args[0]} was not found.",
        ) from error


@router.get("/{model_version}", response_model=ModelDetailRead)
async def model_detail(
    model_version: str, service: ModelServiceDependency
) -> ModelDetailRead:
    try:
        return service.get_model(model_version)
    except ModelNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {error.args[0]} was not found.",
        ) from error


@router.post("/{model_version}/rollback", response_model=RollbackRead)
async def rollback_model(
    model_version: str,
    request: RollbackRequest,
    service: ModelServiceDependency,
) -> RollbackRead:
    try:
        return service.rollback(model_version, request.reason)
    except ModelNotFoundError as error:
        logger.warning("Rollback target does not exist: %s", model_version)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {error.args[0]} was not found.",
        ) from error
    except (ModelRollbackError, ModelUnavailableError) as error:
        logger.warning("Rollback validation failed: target=%s", model_version)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
