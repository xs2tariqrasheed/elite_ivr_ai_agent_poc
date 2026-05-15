"""Phase 10: collect the caller's email address."""

from fastapi import WebSocket

from constants import call_phases as phases

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


async def _run_phase_email(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_EMAIL
    await _speak(websocket, state, [["rec_email"]])
    state.reservation.email = (await _listen(state, max_seconds=15.0)).strip()
    return phases.PHASE_END
