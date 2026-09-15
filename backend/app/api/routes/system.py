from fastapi import APIRouter

from app.api.dependencies import InfrastructureHealthDependency
from app.schemas.training import WorkerHealthResponse

router = APIRouter(prefix="/api/system", tags=["system infrastructure"])


@router.get("/worker-health", response_model=WorkerHealthResponse)
async def worker_health(
    service: InfrastructureHealthDependency,
) -> WorkerHealthResponse:
    return WorkerHealthResponse(**service.check().__dict__)

