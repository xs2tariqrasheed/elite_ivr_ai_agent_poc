"""SQLAlchemy engine, session factory, and declarative base.

The app talks to Supabase Postgres. We prefer DIRECT_URL (session-mode pooler
on port 5432) over the transaction-mode pooled DATABASE_URL, because the pooled
endpoint disables server-side prepared statements and trips up the driver.
"""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()


def _normalize(url: str) -> str:
    """Force the psycopg (v3) driver and drop pgbouncer-only query params."""
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    return url.split("?", 1)[0]


_raw_url = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
if not _raw_url:
    raise RuntimeError("Set DIRECT_URL or DATABASE_URL in the environment / .env")

DATABASE_URL = _normalize(_raw_url)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_db():
    """FastAPI dependency that yields a request-scoped session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
