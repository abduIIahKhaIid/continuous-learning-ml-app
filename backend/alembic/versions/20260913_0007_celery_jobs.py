"""Add durable Celery dispatch, retry, heartbeat, and progress metadata."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "phase8_celery_jobs"
down_revision: str | None = "phase7_monitoring"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.add_column(
            sa.Column("celery_task_id", sa.String(255), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "retry_count", sa.Integer(), server_default="0", nullable=False
            )
        )
        batch_op.add_column(
            sa.Column(
                "progress_stage",
                sa.String(32),
                server_default="queued",
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column("dispatch_error", sa.String(1000), nullable=True)
        )
        batch_op.add_column(
            sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("last_retry_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.create_index(
            "ix_training_runs_celery_task_id",
            ["celery_task_id"],
            unique=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.drop_index("ix_training_runs_celery_task_id")
        batch_op.drop_column("last_retry_at")
        batch_op.drop_column("heartbeat_at")
        batch_op.drop_column("started_at")
        batch_op.drop_column("dispatched_at")
        batch_op.drop_column("dispatch_error")
        batch_op.drop_column("progress_stage")
        batch_op.drop_column("retry_count")
        batch_op.drop_column("celery_task_id")
