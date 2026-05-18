"""Phase 11: log the reservation summary, say goodbye, and hang up."""

import logging

from fastapi import WebSocket

from constants import call_phases as phases

from phase_handlers.speak import _speak
from services.reservation_service import create_reservation


logger = logging.getLogger(__name__)


async def _run_phase_end(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_END
    logger.info(
        "Reservation summary for call %s: %s",
        state.call_sid,
        state.reservation.as_summary_dict(),
    )

    try:
        account_id = (state.account_info or {}).get("id")
        if account_id is None:
            logger.warning(
                "Skipping reservation save for call %s: no account_id on state",
                state.call_sid,
            )
        else:
            r = state.reservation
            reservation = create_reservation(
                account_id=account_id,
                first_name=r.first_name,
                last_name=r.last_name,
                pickup_date=r.pickup_date,
                pickup_time=r.pickup_time,
                pickup_address=r.pickup_address,
                drop_off_address=r.dropoff_address,
            )
            logger.info(
                "Saved reservation id=%s for call %s",
                reservation.id,
                state.call_sid,
            )
    except Exception:
        logger.exception(
            "Failed to save reservation for call %s", state.call_sid
        )

    await _speak(websocket, state, [["rec_good_bye"]])

    return phases.PHASE_HANGUP
