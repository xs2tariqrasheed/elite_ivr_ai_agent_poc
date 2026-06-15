"""Persist a finalized reservation (step 9) to the database.

Maps the free-form, in-call :class:`ReservationSession` onto the typed
:class:`db.models.Reservation` columns: the spoken pickup date/time string is
parsed into ``Date`` + ``Time``. The schema has no confirmation-number column,
so (POC hack) ``first_name`` stores the caller's full name and ``last_name``
stores the confirmation number. Failures are logged (never raised) so a DB
hiccup can't break an otherwise-completed call.
"""
import logging

from dateutil import parser as dateparser

from db.database import SessionLocal
from db.models import Reservation

log = logging.getLogger("voice")


def save_reservation(session) -> bool:
    """Insert a Reservation row from `session`. Returns True on success."""
    if not session.account_id:
        log.error("Cannot save reservation: session has no account_id")
        return False

    missing = [
        name
        for name, val in (
            ("pickup_datetime", session.pickup_datetime),
            ("pickup_address", session.pickup_address),
            ("dropoff_address", session.dropoff_address),
        )
        if not val
    ]
    if missing:
        log.error("Cannot save reservation: missing %s", ", ".join(missing))
        return False

    try:
        pickup = dateparser.parse(session.pickup_datetime, fuzzy=True)
    except (ValueError, OverflowError) as exc:
        log.error("Cannot parse pickup_datetime %r: %s", session.pickup_datetime, exc)
        return False

    # POC hack: the reservations table has no confirmation-number column, so we
    # repurpose existing columns — first_name holds the caller's full name and
    # last_name holds the confirmation number.
    reservation = Reservation(
        account_id=session.account_id,
        first_name=(session.caller_name or "").strip() or "Unknown",
        last_name=session.confirmation_number or "",
        pickup_date=pickup.date(),
        pickup_time=pickup.time(),
        pickup_address=session.pickup_address,
        drop_off_address=session.dropoff_address,
    )
    try:
        with SessionLocal() as db:
            db.add(reservation)
            db.commit()
            log.info(
                "Reservation saved: id=%s account_id=%s %s %s %s %s",
                reservation.id,
                reservation.account_id,
                reservation.first_name,
                reservation.last_name,
                reservation.pickup_date,
                reservation.pickup_time,
            )
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to save reservation: %s", exc)
        return False
