"""Phase: verify the passenger info read back to the caller (yes/no)."""

import asyncio
import logging

from fastapi import WebSocket

from constants import call_phases as phases
from utils.misc import detect_yes_no

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak
from services import llm


logger = logging.getLogger(__name__)


async def _run_phase_passenger_info_verification(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PASSENGER_INFO_VERIFICATION

    account_number = state.account_info.get("account_number")
    await _speak(
        websocket,
        state,
        [
            ["verifications", account_number],
        ],
    )
    text = await _listen(
        state,
        initial_prompt=(
            "The caller is confirming or denying their passenger information. "
            "They will answer with a short yes/no response."
        ),
        hotwords=(
            "yes, no, yeah, nope, correct, incorrect, right, wrong, "
            "confirm, deny, affirmative, negative, sure, okay"
        ),
    )

    logger.info("passenger_info_verification: caller said %r", text)

    answer = detect_yes_no(text)

    if answer is False or answer is None:
        # try to use LLM to classify the response
        answer = await asyncio.to_thread(llm.detect_yes_no_llm_openai, text)

    if answer is False or answer is None:
        return phases.PHASE_HANGUP
    if answer is True:
        return phases.PHASE_PICKUP_DATE

    # Unclear response — default to hanging up rather than guessing.
    logger.info(
        "passenger_info_verification: could not classify response %r — hanging up",
        text,
    )
    return phases.PHASE_HANGUP
