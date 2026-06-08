"""Lists existing customer accounts for the reservation frontend."""
from fastapi import APIRouter

from db import ACCOUNTS

router = APIRouter()


@router.get("/accounts")
async def accounts():
    """Return existing customers with their index so the client can pick a caller."""
    return [
        {"index": i, "name": a["name"], "phone": a["phone"], "email": a["email"]}
        for i, a in enumerate(ACCOUNTS)
    ]
