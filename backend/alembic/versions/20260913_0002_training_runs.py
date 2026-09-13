"""Add the Phase 3 training run registry."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "phase3_training_runs"
down_revision: str | None = "phase2_samples"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("training_batch_id", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("model_path", sa.String(length=1024), nullable=True),
        sa.Column("artifact_checksum", sa.String(length=64), nullable=True),
        sa.Column("artifact_format", sa.String(length=32), nullable=True),
        sa.Column("algorithm", sa.String(length=255), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=True),
        sa.Column("data_selection", sa.JSON(), nullable=True),
        sa.Column("training_sample_count", sa.Integer(), nullable=False),
        sa.Column("train_sample_count", sa.Integer(), nullable=True),
        sa.Column("test_sample_count", sa.Integer(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1_score", sa.Float(), nullable=True),
        sa.Column("roc_auc", sa.Float(), nullable=True),
        sa.Column("confusion_matrix", sa.JSON(), nullable=True),
        sa.Column("random_state", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error_message", sa.String(length=2000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('training', 'completed', 'failed')",
            name="ck_training_runs_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_version"),
        sa.UniqueConstraint("training_batch_id"),
    )


def downgrade() -> None:
    op.drop_table("training_runs")
