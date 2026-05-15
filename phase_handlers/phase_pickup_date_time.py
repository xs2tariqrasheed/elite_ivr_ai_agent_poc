"""Phase 6: collect the pickup date and time."""

import asyncio
import logging

from fastapi import WebSocket

from constants import call_phases as phases
from services import llm

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


logger = logging.getLogger(__name__)


async def _run_phase_pickup_date_time(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PICKUP_DATE_TIME
    await _speak(websocket, state, [["pickup_date_time"]])
    text = (await _listen(state, max_seconds=15.0)).strip()

    date, time = await asyncio.to_thread(llm.extract_pickup_date_time, text)
    state.reservation.pickup_date = date
    state.reservation.pickup_time = time
    logger.info(
        "Captured pickup date/time: date=%s time=%s (from %r)",
        date,
        time,
        text,
    )
    return phases.PHASE_PICKUP_ADDRESS
