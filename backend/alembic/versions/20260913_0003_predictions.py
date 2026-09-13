"""Add active-model metadata and prediction history."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "phase4_predictions"
down_revision: str | None = "phase3_training_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "training_runs",
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("feature_1", sa.Float(), nullable=False),
        sa.Column("feature_2", sa.Float(), nullable=False),
        sa.Column("feature_3", sa.Float(), nullable=False),
        sa.Column("predicted_class", sa.Integer(), nullable=False),
        sa.Column("prediction_probability", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("actual_label", sa.Integer(), nullable=True),
        sa.Column(
            "feedback_received",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "predicted_class IN (0, 1)",
            name="ck_predictions_predicted_class",
        ),
        sa.CheckConstraint(
            "actual_label IS NULL OR actual_label IN (0, 1)",
            name="ck_predictions_actual_label",
        ),
        sa.ForeignKeyConstraint(
            ["model_version"],
            ["training_runs.model_version"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_predictions_model_version"),
        "predictions",
        ["model_version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_predictions_model_version"), table_name="predictions"
    )
    op.drop_table("predictions")
    op.drop_column("training_runs", "is_active")
