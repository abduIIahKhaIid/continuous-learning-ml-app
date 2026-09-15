from fastapi import APIRouter

from app.api.routes.data import router as data_router
from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.monitoring import router as monitoring_router
from app.api.routes.prediction import router as prediction_router
from app.api.routes.training import router as training_router
from app.api.routes.system import router as system_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(data_router)
api_router.include_router(prediction_router)
api_router.include_router(training_router)
api_router.include_router(monitoring_router)
api_router.include_router(models_router)
api_router.include_router(system_router)
