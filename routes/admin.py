"""Admin UI — Jinja2-rendered page for triggering backend APIs."""

import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


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
