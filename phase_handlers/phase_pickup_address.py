"""Phase 7: collect the pickup address."""

from fastapi import WebSocket

from constants import call_phases as phases

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


async def _run_phase_pickup_address(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PICKUP_ADDRESS
    await _speak(websocket, state, [["pickup_address"]])
    pickup_address = await _listen(state, max_seconds=15.0)
    state.reservation.pickup_address = pickup_address.strip()

    return phases.PHASE_DROPOFF_ADDRESS
