"""Phase 11: log the reservation summary, say goodbye, and hang up."""

import logging

from fastapi import WebSocket

from constants import audio_files as audio_const
from constants import call_phases as phases

from phase_handlers.speak import _speak


logger = logging.getLogger(__name__)


async def _run_phase_end(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_END
    logger.info(
        "Reservation summary for call %s: %s",
        state.call_sid,
        state.reservation.as_summary_dict(),
    )
    await _speak(websocket, state, [[audio_const.GOOD_BYE]])
    return phases.PHASE_HANGUP
