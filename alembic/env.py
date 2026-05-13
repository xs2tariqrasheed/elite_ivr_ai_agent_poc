"""Alembic environment.

Reads DATABASE_URL from this project's config.py (which loads .env), and
targets Base.metadata from db.models so autogenerate works out of the box.
"""
import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make the project root importable when alembic runs from the repo root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as app_config  # noqa: E402
from db.database import Base  # noqa: E402
from db import models  # noqa: F401,E402  (register models on Base.metadata)


# Alembic config object — values from alembic.ini
config = context.config

# Resolve the URL Alembic should connect with. Precedence:
#   1. `-x db_url=...` CLI override
#   2. DIRECT_URL env var          (preferred — direct, no pgbouncer)
#   3. DATABASE_URL env var        (fallback if no direct URL is configured)
#   4. config.DIRECT_URL / DATABASE_URL (defaults from config.py)
#
# Migrations must NOT go through pgbouncer in transaction-pool mode; DDL inside
# a transaction breaks under that pool mode. On Supabase: use the port-5432 URL.
_cmd_kwargs = context.get_x_argument(as_dictionary=True)
_resolved_url = (
    _cmd_kwargs.get("db_url")
    or os.environ.get("DIRECT_URL")
    or os.environ.get("DATABASE_URL")
    or app_config.DIRECT_URL
    or app_config.DATABASE_URL
)

# Strip non-DBAPI query params (psycopg2 rejects `pgbouncer=true` etc).
from sqlalchemy.engine.url import make_url  # noqa: E402
_url_obj = make_url(_resolved_url)
_q = dict(_url_obj.query)
for _k in ("pgbouncer", "schema", "connection_limit", "pool_timeout"):
    _q.pop(_k, None)
_resolved_url = str(_url_obj.set(query=_q))

config.set_main_option("sqlalchemy.url", _resolved_url)

# Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits SQL to stdout)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connects to the DB)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
