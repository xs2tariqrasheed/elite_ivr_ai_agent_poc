"""Phase 1: greet the caller and classify their intent."""

import asyncio
import logging
import random
import string

from fastapi import WebSocket

from constants import call_phases as phases
from services import intent_service as intent

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak
from services import llm


logger = logging.getLogger(__name__)


def generate_random_reservation_number() -> str:
    """Return a 6-character id: three uppercase letters then three digits (e.g. AJX123)."""
    letters = "".join(random.choices(string.ascii_uppercase, k=3))
    digits = "".join(random.choices(string.digits, k=3))

    result = f"{letters}{digits}"
    return result


async def _run_phase_intent(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_INTENT
    account_number = state.account_info.get("account_number")
    # generate a random 6-char reservation id (AAA###)
    state.reservation.reservation_number = generate_random_reservation_number()
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

    label = await asyncio.to_thread(llm.classify_intent_openai, text)
    if label == intent.INTENT_NEW_RESERVATION:
        return phases.PHASE_PASSENGER_INFO_VERIFICATION

    # "other" intent — apologize and hang up
    await _speak(websocket, state, [["rec_other_intent"]])
    return phases.PHASE_HANGUP
