from app.schemas.data import DataRequest, DataResponse


def process_data(payload: DataRequest) -> DataResponse:
    """Return a validated payload until persistence is introduced."""
    return DataResponse.model_validate(payload.model_dump())
