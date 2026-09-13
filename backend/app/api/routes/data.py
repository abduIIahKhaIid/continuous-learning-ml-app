from fastapi import APIRouter

from app.schemas.data import DataRequest, DataResponse
from app.services.data import process_data

router = APIRouter(prefix="/api", tags=["data"])


@router.post("/data", response_model=DataResponse)
async def submit_data(payload: DataRequest) -> DataResponse:
    return process_data(payload)
