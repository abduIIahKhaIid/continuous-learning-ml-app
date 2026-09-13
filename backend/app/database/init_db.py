from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, inspect

from app.database.session import engine

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PHASE_2_REVISION = "phase2_samples"
PHASE_3_REVISION = "phase3_training_runs"
PHASE_4_REVISION = "phase4_predictions"
PHASE_5_REVISION = "phase5_feedback_lineage"
HEAD_REVISION = "phase6_continuous_training"


def _alembic_config() -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    return config


def init_db(database_engine: Engine = engine) -> None:
    """Upgrade a new or existing development database to the latest schema."""
    config = _alembic_config()
    adopted_revision: str | None = None
    with database_engine.connect() as connection:
        table_names = set(inspect(connection).get_table_names())
        if "alembic_version" not in table_names and "samples" in table_names:
            if "predictions" in table_names:
                sample_columns = {
                    column["name"]
                    for column in inspect(connection).get_columns("samples")
                }
                adopted_revision = (
                    PHASE_5_REVISION
                    if "source_prediction_id" in sample_columns
                    else PHASE_4_REVISION
                )
            elif "training_runs" in table_names:
                adopted_revision = PHASE_3_REVISION
            else:
                adopted_revision = PHASE_2_REVISION

    if adopted_revision is not None:
        with database_engine.connect() as connection:
            config.attributes["connection"] = connection
            command.stamp(config, adopted_revision)

    with database_engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
