from pathlib import Path

import sqlalchemy as sa
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
            "model_data_profiles",
            "model_events",
            "monitoring_snapshots",
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
        sample_columns = {
            column["name"]
            for column in inspect(engine).get_columns("samples")
        }
        assert "source_prediction_id" in sample_columns
        assert any(
            constraint["column_names"] == ["source_prediction_id"]
            for constraint in inspect(engine).get_unique_constraints("samples")
        )
        assert any(
            foreign_key["constrained_columns"] == ["source_prediction_id"]
            and foreign_key["referred_table"] == "predictions"
            for foreign_key in inspect(engine).get_foreign_keys("samples")
        )
        assert "last_triggered_training_run_id" in sample_columns
        training_run_columns = {
            column["name"]
            for column in inspect(engine).get_columns("training_runs")
        }
        assert {
            "trigger_type",
            "trigger_new_sample_count",
            "trigger_sample_ids",
            "evaluation_sample_ids",
            "active_model_version_before",
            "active_comparison_metrics",
            "rejection_reason",
            "concurrency_slot",
            "promoted_at",
            "completed_at",
            "celery_task_id",
            "retry_count",
            "progress_stage",
            "dispatch_error",
            "dispatched_at",
            "started_at",
            "heartbeat_at",
            "last_retry_at",
        }.issubset(training_run_columns)
        assert any(
            constraint["column_names"] == ["model_version", "feature_name"]
            for constraint in inspect(engine).get_unique_constraints(
                "model_data_profiles"
            )
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
        metadata = sa.MetaData()
        sa.Table(
            "samples",
            metadata,
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("feature_1", sa.Float(), nullable=False),
            sa.Column("feature_2", sa.Float(), nullable=False),
            sa.Column("feature_3", sa.Float(), nullable=False),
            sa.Column("label", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column(
                "used_for_training",
                sa.Boolean(),
                server_default=sa.false(),
                nullable=False,
            ),
            sa.Column("training_batch_id", sa.String(255), nullable=True),
            sa.Column("model_version", sa.String(255), nullable=True),
        ).create(bind=engine)

        init_db(engine)

        assert {"training_runs", "predictions"}.issubset(
            inspect(engine).get_table_names()
        )
    finally:
        engine.dispose()
