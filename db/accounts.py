"""Account lookups backed by the database (see :class:`db.models.Account`).

Returns plain dicts (not ORM instances) so callers can use the data after the
session is closed; downstream consumers read name/email/phone off the dict.
"""
from typing import Optional

from sqlalchemy import select

from db.database import SessionLocal
from db.models import Account
from utils.phone import phones_match


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
    """Return the account whose phone matches `phone`, or None.

    Matching is format-agnostic: a caller arriving as ``+923234251430`` matches
    an account stored as ``03234251430`` or ``+92-323-4251430`` (see
    :func:`utils.phone.phones_match`). The phone column has no canonical form, so
    we scan accounts with a phone set and compare on significant digits.
    """
    if not phone:
        return None
    with SessionLocal() as db:
        accounts = db.scalars(select(Account).where(Account.phone.is_not(None)))
        for account in accounts:
            if phones_match(account.phone, phone):
                return _to_dict(account)
        return None
