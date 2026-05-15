"""add email column to accounts_table

Revision ID: c1f4a2e7b8d3
Revises: bb76c3d49a86
Create Date: 2026-05-15 00:00:00.000000

"""
import secrets
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1f4a2e7b8d3'
down_revision: Union[str, None] = 'bb76c3d49a86'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_EMAIL_DOMAINS = (
    "example.com",
    "mail.example.com",
    "demo.example.net",
    "test.example.org",
)


def _random_email() -> str:
    token = secrets.token_hex(4)
    domain = _EMAIL_DOMAINS[secrets.randbelow(len(_EMAIL_DOMAINS))]
    return f"user_{token}@{domain}"


def upgrade() -> None:
    op.add_column(
        'accounts_table',
        sa.Column('email', sa.String(length=255), nullable=True),
    )

    bind = op.get_bind()
    rows = bind.execute(sa.text("SELECT id FROM accounts_table")).fetchall()
    for row in rows:
        bind.execute(
            sa.text("UPDATE accounts_table SET email = :email WHERE id = :id"),
            {"email": _random_email(), "id": row[0]},
        )

    op.create_index(
        op.f('ix_accounts_table_email'),
        'accounts_table',
        ['email'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_accounts_table_email'), table_name='accounts_table')
    op.drop_column('accounts_table', 'email')
