from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.repositories.data import SqlAlchemyDataRepository


async def get_data_repository(
    session: Annotated[Session, Depends(get_db)],
) -> SqlAlchemyDataRepository:
    return SqlAlchemyDataRepository(session)


DataRepositoryDependency = Annotated[
    SqlAlchemyDataRepository,
    Depends(get_data_repository),
]
