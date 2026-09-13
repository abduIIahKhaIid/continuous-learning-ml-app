from fastapi import APIRouter

from app.api.dependencies import MonitoringServiceDependency
from app.schemas.monitoring import (
    DriftRead,
    MonitoringCheckRead,
    MonitoringHealthRead,
    PerformanceRead,
)

router = APIRouter(prefix="/api/monitoring", tags=["model monitoring"])


@router.get("/drift", response_model=DriftRead)
async def data_drift(service: MonitoringServiceDependency) -> DriftRead:
    return service.data_drift()


@router.get("/performance", response_model=PerformanceRead)
async def model_performance(
    service: MonitoringServiceDependency,
) -> PerformanceRead:
    return service.performance()


@router.get("/health", response_model=MonitoringHealthRead)
async def model_health(
    service: MonitoringServiceDependency,
) -> MonitoringHealthRead:
    return service.health()


@router.post("/check", response_model=MonitoringCheckRead)
async def persist_monitoring_check(
    service: MonitoringServiceDependency,
) -> MonitoringCheckRead:
    return service.check()

