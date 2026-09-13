from app.repositories.data import DataRepository
from app.schemas.data import DataCreate, DataRead


class DataNotFoundError(Exception):
    pass


def create_data(
    payload: DataCreate,
    repository: DataRepository,
) -> DataRead:
    sample = repository.create(**payload.model_dump())
    return DataRead.model_validate(sample)


def list_data(
    *,
    skip: int,
    limit: int,
    repository: DataRepository,
) -> list[DataRead]:
    samples = repository.list(skip=skip, limit=limit)
    return [DataRead.model_validate(sample) for sample in samples]


def get_data(
    record_id: int,
    repository: DataRepository,
) -> DataRead:
    sample = repository.get(record_id)
    if sample is None:
        raise DataNotFoundError(record_id)
    return DataRead.model_validate(sample)
