"""Phase 7: collect the pickup address."""

from fastapi import WebSocket

from constants import audio_files as audio_const
from constants import call_phases as phases

from phase_handlers.call_phases import _listen, _speak


async def _run_phase_pickup_address(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PICKUP_ADDRESS
    await _speak(websocket, state, audio_const.PICKUP_ADDRESS)
    state.reservation.pickup_address = (await _listen(state, max_seconds=15.0)).strip()
    return phases.PHASE_DROPOFF_ADDRESS
