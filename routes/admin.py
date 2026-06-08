"""Passcode-gated Jinja2 admin frontend.

Provides CRUD for accounts and a reservations table (view + delete). Access is
guarded by a single shared passcode (env PASSCODE); a signed session cookie
remembers a successful login.
"""
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from configs.settings import settings
from db.database import get_db
from db.models import Account, Reservation

router = APIRouter(prefix="/admin", tags=["admin"])

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def require_auth(request: Request) -> None:
    """Redirect to the login page unless the session is authenticated."""
    if not request.session.get("authed"):
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": "/admin/login"},
        )


# --- auth ---------------------------------------------------------------


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    if request.session.get("authed"):
        return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login", response_class=HTMLResponse)
async def login(request: Request, passcode: str = Form(...)):
    if not settings.passcode or passcode != settings.passcode:
        return templates.TemplateResponse(
            request, "login.html", {"error": "Incorrect passcode."}, status_code=401
        )
    request.session["authed"] = True
    return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/admin/login", status_code=status.HTTP_303_SEE_OTHER)


# --- accounts CRUD ------------------------------------------------------


@router.get("", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def accounts_list(request: Request, db: Session = Depends(get_db)):
    accounts = db.scalars(select(Account).order_by(Account.id)).all()
    return templates.TemplateResponse(request, "accounts.html", {"accounts": accounts})


@router.get("/accounts/new", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
async def account_new(request: Request):
    return templates.TemplateResponse(
        request, "account_form.html", {"account": None, "error": None}
    )


@router.post("/accounts", dependencies=[Depends(require_auth)])
async def account_create(
    request: Request,
    account_number: str = Form(...),
    name: str = Form(...),
    cid: str = Form(""),
    phone: str = Form(""),
    email: str = Form(""),
    db: Session = Depends(get_db),
):
    account = Account(
        account_number=account_number.strip(),
        name=name.strip(),
        cid=cid.strip() or None,
        phone=phone.strip() or None,
        email=email.strip() or None,
    )
    db.add(account)
    try:
        db.commit()
    except Exception:
        db.rollback()
        return templates.TemplateResponse(
            request,
            "account_form.html",
            {"account": None, "error": "Could not save — account_number must be unique."},
            status_code=400,
        )
    return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)


@router.get(
    "/accounts/{account_id}/edit",
    response_class=HTMLResponse,
    dependencies=[Depends(require_auth)],
)
async def account_edit(request: Request, account_id: int, db: Session = Depends(get_db)):
    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return templates.TemplateResponse(
        request, "account_form.html", {"account": account, "error": None}
    )


@router.post("/accounts/{account_id}", dependencies=[Depends(require_auth)])
async def account_update(
    request: Request,
    account_id: int,
    account_number: str = Form(...),
    name: str = Form(...),
    cid: str = Form(""),
    phone: str = Form(""),
    email: str = Form(""),
    db: Session = Depends(get_db),
):
    account = db.get(Account, account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Account not found")
    account.account_number = account_number.strip()
    account.name = name.strip()
    account.cid = cid.strip() or None
    account.phone = phone.strip() or None
    account.email = email.strip() or None
    try:
        db.commit()
    except Exception:
        db.rollback()
        return templates.TemplateResponse(
            request,
            "account_form.html",
            {"account": account, "error": "Could not save — account_number must be unique."},
            status_code=400,
        )
    return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/accounts/{account_id}/delete", dependencies=[Depends(require_auth)])
async def account_delete(account_id: int, db: Session = Depends(get_db)):
    account = db.get(Account, account_id)
    if account is not None:
        db.delete(account)
        db.commit()
    return RedirectResponse("/admin", status_code=status.HTTP_303_SEE_OTHER)


# --- reservations -------------------------------------------------------


@router.get(
    "/reservations",
    response_class=HTMLResponse,
    dependencies=[Depends(require_auth)],
)
async def reservations_list(request: Request, db: Session = Depends(get_db)):
    reservations = db.scalars(
        select(Reservation)
        .options(selectinload(Reservation.account))
        .order_by(Reservation.pickup_date, Reservation.pickup_time)
    ).all()
    return templates.TemplateResponse(
        request, "reservations.html", {"reservations": reservations}
    )


@router.post("/reservations/{reservation_id}/delete", dependencies=[Depends(require_auth)])
async def reservation_delete(reservation_id: int, db: Session = Depends(get_db)):
    reservation = db.get(Reservation, reservation_id)
    if reservation is not None:
        db.delete(reservation)
        db.commit()
    return RedirectResponse("/admin/reservations", status_code=status.HTTP_303_SEE_OTHER)
