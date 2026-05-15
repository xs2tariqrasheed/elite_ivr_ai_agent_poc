"""Phase: confirm with the caller whether to capture an email (yes/no)."""

import logging

from fastapi import WebSocket

from constants import call_phases as phases
from utils.misc import detect_yes_no

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


logger = logging.getLogger(__name__)


async def _run_phase_email_confirmation(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_EMAIL_CONFIRMATION

    await _speak(websocket, state, [["email_confirmation"]])
    text = await _listen(
        state,
        initial_prompt=(
            "The caller is confirming or denying whether to provide an email. "
            "They will answer with a short yes/no response."
        ),
        hotwords=(
            "yes, no, yeah, nope, correct, incorrect, right, wrong, "
            "confirm, deny, affirmative, negative, sure, okay"
        ),
    )

    logger.info("email_confirmation: caller said %r", text)

    answer = detect_yes_no(text)
    if answer is False:
        return phases.PHASE_HANGUP
    if answer is True:
        return phases.PHASE_CALLBACK_NUMBER

    logger.info(
        "email_confirmation: could not classify response %r — hanging up",
        text,
    )
    return phases.PHASE_HANGUP
