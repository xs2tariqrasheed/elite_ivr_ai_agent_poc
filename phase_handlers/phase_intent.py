"""Phase 1: greet the caller and classify their intent."""

import logging
import time

from fastapi import WebSocket

from constants import audio_files as audio_const
from constants import call_phases as phases
from services import intent_service as intent

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


logger = logging.getLogger(__name__)


async def _run_phase_intent(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_INTENT
    await _speak(
        websocket,
        state,
        [["account_names", "12345"], [audio_const.GREET_UNKNOWN]],
    )
    text = await _listen(
        state,
        initial_prompt="The caller is calling Elite Limousine to make a reservation or ask for help.",
        hotwords=(
            "reservation, booking, car, limousine, limo, vehicle, "
            "ride, pickup, dropoff, chxzauffeur, airport, sedan, "
            "SUV, luxury, transportation, availability, cancel, "
            "modify, reschedule, tonight, tomorrow"
        ),
    )

    start_time = time.time()
    label = intent.classify_with_threshold(text)
    predict_time = time.time() - start_time
    logger.info(f"***** classify_with_threshold: {predict_time}")

    if label == intent.INTENT_NEW_RESERVATION:
        return phases.PHASE_ACCOUNT_NUMBER

    # "other" intent — apologise and hang up
    await _speak(websocket, state, [[audio_const.OTHER_INTENT]])
    return phases.PHASE_HANGUP
