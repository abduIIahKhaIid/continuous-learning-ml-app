from pathlib import Path

from sqlalchemy import inspect, text

from app.database.init_db import HEAD_REVISION, init_db
from app.database.session import create_db_engine
from app.models.sample import Sample


def test_migrations_create_fresh_schema(tmp_path: Path) -> None:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    try:
        init_db(engine)

        assert set(inspect(engine).get_table_names()) == {
            "alembic_version",
            "predictions",
            "samples",
            "training_runs",
        }
        assert "is_active" in {
            column["name"]
            for column in inspect(engine).get_columns("training_runs")
        }
        assert {
            "predicted_class",
            "prediction_probability",
            "model_version",
            "actual_label",
            "feedback_received",
        }.issubset(
            {
                column["name"]
                for column in inspect(engine).get_columns("predictions")
            }
        )
        with engine.connect() as connection:
            revision = connection.scalar(
                text("SELECT version_num FROM alembic_version")
            )
        assert revision == HEAD_REVISION
    finally:
        engine.dispose()


def test_migrations_adopt_existing_phase_two_database(tmp_path: Path) -> None:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    try:
        Sample.__table__.create(bind=engine)

        init_db(engine)

        assert {"training_runs", "predictions"}.issubset(
            inspect(engine).get_table_names()
        )
    finally:
        engine.dispose()
