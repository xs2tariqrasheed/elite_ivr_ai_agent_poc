"""Admin UI — Jinja2-rendered page for triggering backend APIs."""

import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from db.database import get_db
from db.models import Account, Reservation


_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "templates",
)
templates = Jinja2Templates(directory=_TEMPLATES_DIR)

router = APIRouter()


@router.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request) -> HTMLResponse:
    """Render the admin console for invoking backend APIs."""
    return templates.TemplateResponse(
        "admin.html",
        {"request": request, "title": "Admin Console"},
    )


@router.get("/admin-reservations", response_class=HTMLResponse)
def admin_reservations_page(request: Request) -> HTMLResponse:
    """Render the read-only admin page listing accounts and reservations."""
    with get_db() as session:
        accounts = session.execute(
            select(Account).order_by(Account.id)
        ).scalars().all()
        reservations = session.execute(
            select(Reservation).order_by(Reservation.id)
        ).scalars().all()

        accounts_view = [
            {
                "id": a.id,
                "account_number": a.account_number,
                "name": a.name,
                "cid": a.cid,
                "phone": a.phone,
            }
            for a in accounts
        ]
        reservations_view = [
            {
                "id": r.id,
                "account_id": r.account_id,
                "first_name": r.first_name,
                "last_name": r.last_name,
                "pickup_date": r.pickup_date.isoformat() if r.pickup_date else "",
                "pickup_time": r.pickup_time.isoformat() if r.pickup_time else "",
                "pickup_address": r.pickup_address,
                "drop_off_address": r.drop_off_address,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in reservations
        ]

    return templates.TemplateResponse(
        "admin_reservations.html",
        {
            "request": request,
            "title": "Accounts & Reservations",
            "accounts": accounts_view,
            "reservations": reservations_view,
        },
    )
