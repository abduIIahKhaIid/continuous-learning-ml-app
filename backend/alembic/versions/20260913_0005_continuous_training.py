"""Add continuous-training state, audit metadata, and trigger checkpoints."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "phase6_continuous_training"
down_revision: str | None = "phase5_feedback_lineage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.drop_constraint("ck_training_runs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_training_runs_status",
            "status IN ('queued', 'running', 'training', 'completed', "
            "'failed', 'rejected', 'promoted')",
        )
        batch_op.add_column(
            sa.Column("trigger_type", sa.String(length=32), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "trigger_new_sample_count",
                sa.Integer(),
                server_default="0",
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column("trigger_sample_ids", sa.JSON(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("evaluation_sample_ids", sa.JSON(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "active_model_version_before",
                sa.String(length=64),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column("active_comparison_metrics", sa.JSON(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "rejection_reason", sa.String(length=2000), nullable=True
            )
        )
        batch_op.add_column(
            sa.Column("concurrency_slot", sa.String(length=32), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "promoted_at", sa.DateTime(timezone=True), nullable=True
            )
        )
        batch_op.add_column(
            sa.Column(
                "completed_at", sa.DateTime(timezone=True), nullable=True
            )
        )
        batch_op.create_unique_constraint(
            "uq_training_runs_concurrency_slot", ["concurrency_slot"]
        )

    with op.batch_alter_table("samples") as batch_op:
        batch_op.add_column(
            sa.Column(
                "last_triggered_training_run_id",
                sa.Integer(),
                nullable=True,
            )
        )
        batch_op.create_foreign_key(
            "fk_samples_last_triggered_training_run_id",
            "training_runs",
            ["last_triggered_training_run_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_index(
            "ix_samples_last_triggered_training_run_id",
            ["last_triggered_training_run_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("samples") as batch_op:
        batch_op.drop_index("ix_samples_last_triggered_training_run_id")
        batch_op.drop_constraint(
            "fk_samples_last_triggered_training_run_id",
            type_="foreignkey",
        )
        batch_op.drop_column("last_triggered_training_run_id")

    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.drop_constraint(
            "uq_training_runs_concurrency_slot", type_="unique"
        )
        batch_op.drop_column("completed_at")
        batch_op.drop_column("promoted_at")
        batch_op.drop_column("concurrency_slot")
        batch_op.drop_column("rejection_reason")
        batch_op.drop_column("active_comparison_metrics")
        batch_op.drop_column("active_model_version_before")
        batch_op.drop_column("evaluation_sample_ids")
        batch_op.drop_column("trigger_sample_ids")
        batch_op.drop_column("trigger_new_sample_count")
        batch_op.drop_column("trigger_type")
        batch_op.drop_constraint("ck_training_runs_status", type_="check")
        batch_op.create_check_constraint(
            "ck_training_runs_status",
            "status IN ('training', 'completed', 'failed')",
        )
