"""Phase 1: greet the caller and classify their intent."""

import logging
import asyncio

from fastapi import WebSocket

from constants import call_phases as phases
from services import intent_service as intent

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak
from services import llm


logger = logging.getLogger(__name__)


async def _run_phase_intent(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_INTENT
    account_number = state.account_info.get("account_number")
    await _speak(
        websocket,
        state,
        [["known_greet_hi", account_number], ["greet"]],
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

    label = intent.classify_with_threshold(text)

    if label == intent.INTENT_NEW_RESERVATION:
        return phases.PHASE_PASSENGER_INFO_VERIFICATION

    label = await asyncio.to_thread(llm.classify_intent, text)
    if label == intent.INTENT_NEW_RESERVATION:
        return phases.PHASE_PASSENGER_INFO_VERIFICATION

    # "other" intent — apologize and hang up
    await _speak(websocket, state, [["rec_other_intent"]])
    return phases.PHASE_HANGUP
