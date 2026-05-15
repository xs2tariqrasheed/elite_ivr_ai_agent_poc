"""Phase 3: collect the account-holder's name."""

from fastapi import WebSocket

from constants import call_phases as phases

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


async def _run_phase_account_name(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_ACCOUNT_NAME
    await _speak(websocket, state, [["rec_account_name"]])
    state.reservation.account_name = (await _listen(state)).strip()
    return phases.PHASE_FIRST_NAME
