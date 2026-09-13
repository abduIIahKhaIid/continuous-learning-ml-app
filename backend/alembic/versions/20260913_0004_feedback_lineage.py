"""Add immutable feedback lineage to training samples."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "phase5_feedback_lineage"
down_revision: str | None = "phase4_predictions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("samples") as batch_op:
        batch_op.add_column(
            sa.Column("source_prediction_id", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_samples_source_prediction_id_predictions",
            "predictions",
            ["source_prediction_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch_op.create_unique_constraint(
            "uq_samples_source_prediction_id",
            ["source_prediction_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("samples") as batch_op:
        batch_op.drop_constraint(
            "uq_samples_source_prediction_id", type_="unique"
        )
        batch_op.drop_constraint(
            "fk_samples_source_prediction_id_predictions",
            type_="foreignkey",
        )
        batch_op.drop_column("source_prediction_id")
