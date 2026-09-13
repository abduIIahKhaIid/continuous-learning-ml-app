"""Create the Phase 2 samples table."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "phase2_samples"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "samples",
        sa.Column("id", sa.Integer(), nullable=False),
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
        sa.Column("training_batch_id", sa.String(length=255), nullable=True),
        sa.Column("model_version", sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("samples")
