"""Account service.

Combines:
  * legacy dummy-JSON lookup used by the live IVR flow
    (``load_accounts`` / ``find_account_by_number``)
  * new SQLAlchemy/Postgres-backed CRUD helpers
    (``create_account`` / ``update_account`` / ``delete_account`` and
    a couple of read helpers)
"""
import json
import logging
import os
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.orm import Session

import config
from db.database import get_db
from db.models import Account

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy dummy-data lookup (kept so the IVR call flow keeps working)
# ---------------------------------------------------------------------------

_accounts: List[Dict] = []


def load_accounts() -> None:
    """Read ``DUMMY_DATA_PATH`` once at startup."""
    global _accounts
    path = config.DUMMY_DATA_PATH
    if not os.path.exists(path):
        logger.warning("Dummy data file not found at %s — accounts list will be empty", path)
        _accounts = []
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            _accounts = json.load(f)
        logger.info("Loaded %d dummy accounts from %s", len(_accounts), path)
    except Exception:
        logger.exception("Failed to load dummy accounts from %s", path)
        _accounts = []


def find_account_by_number(account_number: str) -> Optional[Dict]:
    """Return the account dict for ``account_number`` or ``None``."""
    if not account_number:
        return None
    for acc in _accounts:
        if str(acc.get("account_number")) == str(account_number):
            return acc
    return None


# ---------------------------------------------------------------------------
# DB-backed CRUD
# ---------------------------------------------------------------------------


class AccountNotFoundError(Exception):
    """Raised when an account lookup by id / account_number fails."""


class AccountAlreadyExistsError(Exception):
    """Raised when trying to create an account with a duplicate ``account_number``."""


def _get_account_or_raise(db: Session, account_id: int) -> Account:
    account = db.get(Account, account_id)
    if account is None:
        raise AccountNotFoundError(f"Account id={account_id} not found")
    return account


def create_account(
    *,
    account_number: str,
    name: str,
    cid: Optional[str] = None,
    phone: Optional[str] = None,
    db: Optional[Session] = None,
) -> Account:
    """Create a new account.

    If ``db`` is provided the caller is responsible for the session lifecycle;
    otherwise a session is opened, committed, and closed inside this function.
    """
    def _do(session: Session) -> Account:
        account = Account(
            account_number=account_number,
            name=name,
            cid=cid,
            phone=phone,
        )
        session.add(account)
        try:
            session.flush()
        except IntegrityError as exc:
            session.rollback()
            raise AccountAlreadyExistsError(
                f"Account with account_number={account_number!r} already exists"
            ) from exc
        logger.info("Created account id=%s account_number=%s", account.id, account.account_number)
        return account

    if db is not None:
        return _do(db)

    with get_db() as session:
        account = _do(session)
        session.commit()
        session.refresh(account)
        return account


def update_account(
    account_id: int,
    *,
    account_number: Optional[str] = None,
    name: Optional[str] = None,
    cid: Optional[str] = None,
    phone: Optional[str] = None,
    db: Optional[Session] = None,
) -> Account:
    """Update fields on an existing account. Only provided fields are changed."""
    def _do(session: Session) -> Account:
        account = _get_account_or_raise(session, account_id)
        if account_number is not None:
            account.account_number = account_number
        if name is not None:
            account.name = name
        if cid is not None:
            account.cid = cid
        if phone is not None:
            account.phone = phone
        try:
            session.flush()
        except IntegrityError as exc:
            session.rollback()
            raise AccountAlreadyExistsError(
                f"Cannot update — account_number={account_number!r} already in use"
            ) from exc
        logger.info("Updated account id=%s", account.id)
        return account

    if db is not None:
        return _do(db)

    with get_db() as session:
        account = _do(session)
        session.commit()
        session.refresh(account)
        return account


def delete_account(account_id: int, *, db: Optional[Session] = None) -> None:
    """Delete an account (and, by cascade, its reservations)."""
    def _do(session: Session) -> None:
        account = _get_account_or_raise(session, account_id)
        session.delete(account)
        logger.info("Deleted account id=%s", account_id)

    if db is not None:
        _do(db)
        return

    with get_db() as session:
        _do(session)
        session.commit()


def get_account(account_id: int, *, db: Optional[Session] = None) -> Account:
    """Fetch an account by primary key (raises if not found)."""
    if db is not None:
        return _get_account_or_raise(db, account_id)
    with get_db() as session:
        return _get_account_or_raise(session, account_id)


def get_account_by_account_number(
    account_number: str, *, db: Optional[Session] = None
) -> Optional[Account]:
    """Fetch an account by its business ``account_number`` (or ``None``)."""
    stmt = select(Account).where(Account.account_number == account_number)
    if db is not None:
        return db.execute(stmt).scalar_one_or_none()
    with get_db() as session:
        return session.execute(stmt).scalar_one_or_none()
