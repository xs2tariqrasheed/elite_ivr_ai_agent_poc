"""Phase 5: collect the passenger's last name."""

from fastapi import WebSocket

from constants import audio_files as audio_const
from constants import call_phases as phases

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


async def _run_phase_last_name(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_LAST_NAME
    await _speak(websocket, state, audio_const.LAST_NAME)
    state.reservation.last_name = (await _listen(state)).strip()
    return phases.PHASE_PICKUP_DATE_TIME
