from fastapi import APIRouter

from app.api.routes.data import router as data_router
from app.api.routes.health import router as health_router
from app.api.routes.prediction import router as prediction_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(data_router)
api_router.include_router(prediction_router)
