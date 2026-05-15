"""Phase 6: collect the pickup date and time."""

from fastapi import WebSocket

from constants import call_phases as phases

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


async def _run_phase_pickup_date_time(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PICKUP_DATE_TIME
    await _speak(websocket, state, [["rec_pickup_date_time"]])
    state.reservation.pickup_date_time = (
        await _listen(state, max_seconds=15.0)
    ).strip()
    return phases.PHASE_PICKUP_ADDRESS
