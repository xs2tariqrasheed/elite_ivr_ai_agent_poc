"""SQLAlchemy engine and session setup for Postgres."""
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Tuple

from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL, make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool

import config

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# Query-string keys that are markers for us / Supabase / Prisma but that psycopg2
# does not understand and will reject as invalid DSN options.
_NON_DBAPI_QUERY_KEYS = {"pgbouncer", "schema", "connection_limit", "pool_timeout"}


def _normalize_url(raw_url: str) -> Tuple[URL, bool]:
    """Parse the URL, strip query params psycopg2 can't pass through to libpq,
    and return (clean URL object, is_pgbouncer flag).
    """
    url = make_url(raw_url)
    query = dict(url.query)

    is_pgbouncer = (
        query.get("pgbouncer", "").lower() == "true"
        or url.port == 6543
    )

    # Drop keys psycopg2 will reject.
    for k in _NON_DBAPI_QUERY_KEYS:
        query.pop(k, None)
    url = url.set(query=query)
    return url, is_pgbouncer


def _build_engine_kwargs(is_pgbouncer: bool, url: URL) -> Dict[str, Any]:
    """Pick engine settings.

    If the URL points at pgbouncer (Supabase port 6543, ``?pgbouncer=true``),
    we disable SQLAlchemy's own connection pool — pgbouncer is already pooling —
    and (on psycopg v3) we turn off prepared statements so transaction-pool
    mode behaves.
    """
    kwargs: Dict[str, Any] = {
        "echo": config.DB_ECHO,
        "pool_pre_ping": True,
        "future": True,
    }
    if is_pgbouncer:
        kwargs["poolclass"] = NullPool
        # psycopg v3 only — psycopg2 doesn't accept (or need) this kwarg.
        driver = url.get_dialect().driver if url.drivername else ""
        if driver.startswith("psycopg") and driver != "psycopg2":
            kwargs["connect_args"] = {"prepare_threshold": None}
    return kwargs


_clean_url, _is_pgbouncer = _normalize_url(config.DATABASE_URL)

# Single shared engine for the whole app.
engine = create_engine(_clean_url, **_build_engine_kwargs(_is_pgbouncer, _clean_url))

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    future=True,
)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context-managed DB session.

    Usage:
        with get_db() as db:
            ...
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables that don't yet exist.

    Useful for local dev / POC. In production you would use Alembic migrations.
    """
    # Import models so they're registered on Base.metadata before create_all.
    from db import models  # noqa: F401

    logger.info("Creating database tables (if missing) on %s", engine.url)
    Base.metadata.create_all(bind=engine)
