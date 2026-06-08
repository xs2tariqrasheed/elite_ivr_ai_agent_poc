"""Lists existing customer accounts for the reservation frontend."""
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import Account

router = APIRouter()


@router.get("/accounts")
async def accounts(db: Session = Depends(get_db)):
    """Return existing customers with their id so the client can pick a caller."""
    rows = db.scalars(select(Account).order_by(Account.id)).all()
    return [
        {"id": a.id, "name": a.name, "phone": a.phone, "email": a.email}
        for a in rows
    ]
