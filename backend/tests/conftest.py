import os
from collections.abc import Iterator
from pathlib import Path

# Application settings are loaded during test collection. This bootstrap URL
# cannot create a file, and endpoint fixtures still override every DB session
# with a fresh temporary SQLite database.
os.environ.setdefault("DATABASE_URL", "sqlite://")

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.base import Base
from app.database.session import create_db_engine
from app.models import Sample, TrainingRun  # noqa: F401


@pytest.fixture
def test_engine(tmp_path: Path) -> Iterator[Engine]:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(test_engine: Engine) -> Iterator[Session]:
    session_factory = sessionmaker(
        bind=test_engine,
        autoflush=False,
        expire_on_commit=False,
    )
    with session_factory() as session:
        yield session
