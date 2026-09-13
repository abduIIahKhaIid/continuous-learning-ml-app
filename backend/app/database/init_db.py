from sqlalchemy import Engine

from app.database.base import Base
from app.database.session import engine
from app.models.sample import Sample  # noqa: F401


def init_db(database_engine: Engine = engine) -> None:
    """Create development tables; migrations can replace this later."""
    Base.metadata.create_all(bind=database_engine)
