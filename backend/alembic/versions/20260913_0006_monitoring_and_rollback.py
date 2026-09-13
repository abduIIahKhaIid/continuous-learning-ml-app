"""Add data profiles, monitoring snapshots, and model audit events."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "phase7_monitoring"
down_revision: str | None = "phase6_continuous_training"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_data_profiles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("feature_name", sa.String(64), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("mean", sa.Float(), nullable=False),
        sa.Column("std", sa.Float(), nullable=False),
        sa.Column("min", sa.Float(), nullable=False),
        sa.Column("max", sa.Float(), nullable=False),
        sa.Column("median", sa.Float(), nullable=False),
        sa.Column("q25", sa.Float(), nullable=False),
        sa.Column("q75", sa.Float(), nullable=False),
        sa.Column("histogram_bins", sa.JSON(), nullable=False),
        sa.Column("histogram_counts", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["model_version"],
            ["training_runs.model_version"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "model_version",
            "feature_name",
            name="uq_model_profile_feature",
        ),
    )
    op.create_index(
        "ix_model_data_profiles_model_version",
        "model_data_profiles",
        ["model_version"],
    )

    op.create_table(
        "monitoring_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_version", sa.String(64), nullable=True),
        sa.Column("snapshot_type", sa.String(32), nullable=False),
        sa.Column("overall_status", sa.String(32), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["model_version"],
            ["training_runs.model_version"],
            ondelete="RESTRICT",
        ),
    )
    op.create_index(
        "ix_monitoring_snapshots_model_version",
        "monitoring_snapshots",
        ["model_version"],
    )

    op.create_table(
        "model_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column(
            "previous_model_version", sa.String(64), nullable=True
        ),
        sa.Column("reason", sa.String(500), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["model_version"],
            ["training_runs.model_version"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["previous_model_version"],
            ["training_runs.model_version"],
            ondelete="RESTRICT",
        ),
    )
    op.create_index(
        "ix_model_events_model_version",
        "model_events",
        ["model_version"],
    )


def downgrade() -> None:
    op.drop_index("ix_model_events_model_version", table_name="model_events")
    op.drop_table("model_events")
    op.drop_index(
        "ix_monitoring_snapshots_model_version",
        table_name="monitoring_snapshots",
    )
    op.drop_table("monitoring_snapshots")
    op.drop_index(
        "ix_model_data_profiles_model_version",
        table_name="model_data_profiles",
    )
    op.drop_table("model_data_profiles")
