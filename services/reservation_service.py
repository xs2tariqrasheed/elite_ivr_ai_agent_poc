"""Reservation service: DB-backed CRUD helpers."""
import logging
from datetime import date, datetime, time
from typing import Optional, Union

from sqlalchemy.orm import Session

from db.database import get_db
from db.models import Reservation
from services.account_service import AccountNotFoundError, _get_account_or_raise

logger = logging.getLogger(__name__)


def _coerce_date(value: Union[str, date, datetime]) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        # Accept ISO-8601 ('YYYY-MM-DD' or full datetime) strings.
        return datetime.fromisoformat(value).date() if "T" in value or " " in value \
            else date.fromisoformat(value)
    raise TypeError(f"Unsupported pickup_date type: {type(value)!r}")


def _coerce_time(value: Union[str, time, datetime]) -> time:
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, time):
        return value
    if isinstance(value, str):
        # Accept 'HH:MM' or 'HH:MM:SS'.
        return time.fromisoformat(value)
    raise TypeError(f"Unsupported pickup_time type: {type(value)!r}")


def create_reservation(
    *,
    account_id: int,
    first_name: str,
    last_name: str,
    pickup_date: Union[str, date, datetime],
    pickup_time: Union[str, time, datetime],
    pickup_address: str,
    drop_off_address: str,
    reservation_number: Optional[str] = None,
    db: Optional[Session] = None,
) -> Reservation:
    """Create a new reservation for an existing account.

    Raises ``AccountNotFoundError`` if ``account_id`` doesn't exist.
    """
    pd = _coerce_date(pickup_date)
    pt = _coerce_time(pickup_time)

    def _do(session: Session) -> Reservation:
        # Validate the FK target up-front so we get a clear error.
        _get_account_or_raise(session, account_id)

        reservation = Reservation(
            account_id=account_id,
            reservation_number=reservation_number,
            first_name=first_name,
            last_name=last_name,
            pickup_date=pd,
            pickup_time=pt,
            pickup_address=pickup_address,
            drop_off_address=drop_off_address,
        )
        session.add(reservation)
        session.flush()
        logger.info(
            "Created reservation id=%s account_id=%s reservation_number=%s pickup=%s %s",
            reservation.id,
            account_id,
            reservation_number,
            pd,
            pt,
        )
        return reservation

    if db is not None:
        return _do(db)

    with get_db() as session:
        reservation = _do(session)
        session.commit()
        session.refresh(reservation)
        return reservation
