"""Phase 4: collect the passenger's first name."""

from fastapi import WebSocket

from constants import call_phases as phases

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


async def _run_phase_first_name(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_FIRST_NAME
    await _speak(websocket, state, [["rec_first_name"]])
    state.reservation.first_name = (await _listen(state)).strip()
    return phases.PHASE_LAST_NAME
