"""Admin UI — Jinja2-rendered page for triggering backend APIs."""

import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
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


def _format_date(d) -> str:
    """Format a date as e.g. 'May 20, 2026'."""
    if not d:
        return ""
    return d.strftime("%B %-d, %Y")


def _format_time(t) -> str:
    """Format a time as e.g. '2:30 PM'."""
    if not t:
        return ""
    return t.strftime("%-I:%M %p")


def _format_datetime(dt) -> str:
    """Format a datetime as e.g. 'May 20, 2026, 2:30 PM'."""
    if not dt:
        return ""
    return dt.strftime("%B %-d, %Y, %-I:%M %p")


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
                "pickup_date": _format_date(r.pickup_date),
                "pickup_time": _format_time(r.pickup_time),
                "pickup_address": r.pickup_address,
                "drop_off_address": r.drop_off_address,
                "created_at": _format_datetime(r.created_at),
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


@router.delete("/admin-reservations/{reservation_id}")
def delete_reservation(reservation_id: int) -> JSONResponse:
    """Delete a reservation by id."""
    with get_db() as session:
        reservation = session.get(Reservation, reservation_id)
        if reservation is None:
            raise HTTPException(status_code=404, detail="Reservation not found")
        session.delete(reservation)
        session.commit()
    return JSONResponse({"ok": True, "id": reservation_id})
