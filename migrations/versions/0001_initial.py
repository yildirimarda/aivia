"""Create the empty baseline schema.

Revision ID: 0001
Revises:
Create Date: 2026-10-28
"""

from __future__ import annotations

revision: str = "0001"
down_revision: str | None = None
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Establish the baseline; later revisions add the tables."""


def downgrade() -> None:
    """Return to the empty database."""
