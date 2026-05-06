"""Phase 8: collect the drop-off address."""

from fastapi import WebSocket

from constants import audio_files as audio_const
from constants import call_phases as phases

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


async def _run_phase_dropoff_address(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_DROPOFF_ADDRESS
    await _speak(websocket, state, audio_const.DROPOFF_ADDRESS)
    state.reservation.dropoff_address = (await _listen(state, max_seconds=15.0)).strip()
    return phases.PHASE_CALLBACK_NUMBER
