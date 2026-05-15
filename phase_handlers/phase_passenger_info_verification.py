"""Phase: verify the passenger info read back to the caller (yes/no)."""

import logging
import re

from fastapi import WebSocket

from constants import call_phases as phases

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


logger = logging.getLogger(__name__)


_YES_PATTERN = re.compile(
    r"\b(yes|yeah|yep|yup|correct|right|sure|affirmative|"
    r"absolutely|of course|that's right|thats right|ok|okay|confirm|confirmed)\b",
    re.IGNORECASE,
)
_NO_PATTERN = re.compile(
    r"\b(no|nope|nah|negative|incorrect|wrong|"
    r"that's wrong|thats wrong|not right|don't|dont)\b",
    re.IGNORECASE,
)


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

    if _NO_PATTERN.search(text):
        return phases.PHASE_HANGUP
    if _YES_PATTERN.search(text):
        return phases.PHASE_ACCOUNT_NUMBER

    # Unclear response — default to hanging up rather than guessing.
    logger.info(
        "passenger_info_verification: could not classify response %r — hanging up",
        text,
    )
    return phases.PHASE_HANGUP
