"""Phase 11: log the reservation summary, say goodbye, and hang up."""

import asyncio
import logging

from datetime import datetime
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
        name = (state.account_info or {}).get("name")
        date = state.reservation.pickup_date or datetime.now().date()
        time = state.reservation.pickup_time or datetime.now().time()
        if account_id is None:
            logger.warning(
                "Skipping reservation save for call %s: no account_id on state",
                state.call_sid,
            )
        else:
            r = state.reservation
            call_sid = state.call_sid

            def _save_reservation_in_background() -> None:
                try:
                    reservation = create_reservation(
                        account_id=account_id,
                        first_name=name,
                        last_name=r.reservation_number,
                        pickup_date=date,
                        pickup_time=time,
                        pickup_address=r.pickup_address,
                        drop_off_address=r.dropoff_address,
                    )
                    logger.info(
                        "Saved reservation id=%s for call %s",
                        reservation.id,
                        call_sid,
                    )
                except Exception:
                    logger.exception("Failed to save reservation for call %s", call_sid)

            asyncio.create_task(asyncio.to_thread(_save_reservation_in_background))
    except Exception:
        logger.exception("Failed to save reservation for call %s", state.call_sid)

    await _speak(websocket, state, [["reservation_id_message"]])

    return phases.PHASE_HANGUP
