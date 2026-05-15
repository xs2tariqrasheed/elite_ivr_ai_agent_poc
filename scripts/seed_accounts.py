"""Seed the ``accounts_table`` with 10 sample accounts.

Idempotent: if an account_number already exists, the row is updated in place
rather than duplicated. Safe to run multiple times.

Usage (from the project root, with .venv active):
    python -m scripts.seed_accounts
    # or
    python scripts/seed_accounts.py

The script connects using config.DATABASE_URL (your pooled Supabase URL is
fine — this is just regular DML, not DDL).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List

# Allow running as a plain file: `python scripts/seed_accounts.py`
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from db.database import get_db  # noqa: E402
from services.account_service import (  # noqa: E402
    AccountAlreadyExistsError,
    create_account,
    get_account_by_account_number,
    update_account,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed_accounts")


# 10 sample accounts. First four mirror the legacy dummy_data.json so the IVR
# flow keeps recognising the same account numbers used in dev.
SEED_ACCOUNTS: List[Dict[str, str]] = [
    {"account_number": "5678", "name": "Beta Industries",   "cid": "CID-5678", "phone": "+1-555-0202", "email": "contact@beta-industries.example.com"},
    {"account_number": "9012", "name": "Gamma Holdings",    "cid": "CID-9012", "phone": "+1-555-0303", "email": "hello@gamma-holdings.example.com"},
    {"account_number": "3456", "name": "Delta Labs",        "cid": "CID-3456", "phone": "+1-555-0404", "email": "team@delta-labs.example.com"},
    {"account_number": "7890", "name": "Epsilon Partners",  "cid": "CID-7890", "phone": "+1-555-0505", "email": "info@epsilon-partners.example.com"},
    {"account_number": "1234", "name": "Alpha Corp",        "cid": "CID-1234", "phone": "+1-555-0101", "email": "ops@alpha-corp.example.com"},
    {"account_number": "2468", "name": "Zeta Logistics",    "cid": "CID-2468", "phone": "+1-555-0606", "email": "dispatch@zeta-logistics.example.com"},
    {"account_number": "1357", "name": "Eta Ventures",      "cid": "CID-1357", "phone": "+1-555-0707", "email": "hi@eta-ventures.example.com"},
    {"account_number": "8642", "name": "Theta Group",       "cid": "CID-8642", "phone": "+1-555-0808", "email": "support@theta-group.example.com"},
    {"account_number": "9753", "name": "Iota Capital",      "cid": "CID-9753", "phone": "+1-555-0909", "email": "desk@iota-capital.example.com"},
    {"account_number": "1122", "name": "Kappa Travel Co.",  "cid": "CID-1122", "phone": "+1-555-1010", "email": "bookings@kappa-travel.example.com"},
]


def seed() -> None:
    created = updated = 0
    with get_db() as db:
        for row in SEED_ACCOUNTS:
            existing = get_account_by_account_number(row["account_number"], db=db)
            if existing is None:
                try:
                    acct = create_account(db=db, **row)
                    db.flush()
                    logger.info("CREATED  id=%s account_number=%s name=%s",
                                acct.id, acct.account_number, acct.name)
                    created += 1
                except AccountAlreadyExistsError:
                    # Race between check and insert — fall through to update.
                    existing = get_account_by_account_number(row["account_number"], db=db)
                    if existing is None:
                        raise
                    update_account(existing.id, db=db, **{k: v for k, v in row.items()
                                                          if k != "account_number"})
                    logger.info("UPDATED  id=%s account_number=%s",
                                existing.id, existing.account_number)
                    updated += 1
            else:
                update_account(
                    existing.id,
                    db=db,
                    name=row["name"],
                    cid=row["cid"],
                    phone=row["phone"],
                    email=row["email"],
                )
                logger.info("UPDATED  id=%s account_number=%s name=%s",
                            existing.id, existing.account_number, row["name"])
                updated += 1

        db.commit()

    logger.info("Done. created=%d updated=%d total=%d", created, updated, created + updated)


if __name__ == "__main__":
    seed()
