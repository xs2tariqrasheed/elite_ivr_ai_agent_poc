"""add reservation_number to reservations

Revision ID: 2f5b3af569eb
Revises: bb76c3d49a86
Create Date: 2026-05-19 07:28:41.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2f5b3af569eb'
down_revision: Union[str, None] = 'bb76c3d49a86'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'reservations',
        sa.Column('reservation_number', sa.String(length=64), nullable=True),
    )
    op.create_index(
        op.f('ix_reservations_reservation_number'),
        'reservations',
        ['reservation_number'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_reservations_reservation_number'),
        table_name='reservations',
    )
    op.drop_column('reservations', 'reservation_number')
