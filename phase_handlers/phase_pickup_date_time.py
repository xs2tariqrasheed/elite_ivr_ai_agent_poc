"""Phase 6: collect the pickup date and time."""

from fastapi import WebSocket

from constants import audio_files as audio_const
from constants import call_phases as phases

from phase_handlers.call_phases import _listen, _speak


async def _run_phase_pickup_date_time(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PICKUP_DATE_TIME
    await _speak(websocket, state, audio_const.PICKUP_DATE_TIME)
    state.reservation.pickup_date_time = (
        await _listen(state, max_seconds=15.0)
    ).strip()
    return phases.PHASE_PICKUP_ADDRESS
