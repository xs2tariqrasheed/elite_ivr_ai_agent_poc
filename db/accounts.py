"""Account lookups backed by the database (see :class:`db.models.Account`).

Returns plain dicts (not ORM instances) so callers can use the data after the
session is closed; downstream consumers read name/email/phone off the dict.
"""
from typing import Optional

from sqlalchemy import select

from db.database import SessionLocal
from db.models import Account


def _to_dict(account: Account) -> dict:
    return {
        "id": account.id,
        "account_number": account.account_number,
        "name": account.name,
        "cid": account.cid,
        "phone": account.phone,
        "email": account.email,
    }


def get_account(account_id: int) -> Optional[dict]:
    """Return the account with primary key `account_id`, or None."""
    with SessionLocal() as db:
        account = db.get(Account, account_id)
        return _to_dict(account) if account else None


def get_account_by_phone(phone: str) -> Optional[dict]:
    """Return the account whose phone matches `phone` (E.164), or None."""
    with SessionLocal() as db:
        account = db.scalars(select(Account).where(Account.phone == phone)).first()
        return _to_dict(account) if account else None
