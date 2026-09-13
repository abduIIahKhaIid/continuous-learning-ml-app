from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect

from app.database.session import engine

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PHASE_2_REVISION = "phase2_samples"
PHASE_3_REVISION = "phase3_training_runs"
HEAD_REVISION = "phase4_predictions"


def _alembic_config() -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    return config


def init_db(database_engine: Engine = engine) -> None:
    """Upgrade a new or existing development database to the latest schema."""
    config = _alembic_config()
    with database_engine.begin() as connection:
        table_names = set(inspect(connection).get_table_names())
        config.attributes["connection"] = connection

        if "alembic_version" not in table_names and "samples" in table_names:
            if "predictions" in table_names:
                adopted_revision = HEAD_REVISION
            elif "training_runs" in table_names:
                adopted_revision = PHASE_3_REVISION
            else:
                adopted_revision = PHASE_2_REVISION
            command.stamp(config, adopted_revision)

        command.upgrade(config, "head")
