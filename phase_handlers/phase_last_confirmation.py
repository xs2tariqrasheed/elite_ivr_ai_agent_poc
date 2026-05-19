"""Phase: last confirmation."""

import asyncio
import logging

from fastapi import WebSocket

from constants import call_phases as phases

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak

from services import llm
from utils.misc import detect_yes_no


logger = logging.getLogger(__name__)


async def _run_phase_last_confirmation(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_LAST_CONFIRMATION
    await _speak(websocket, state, [["last_confirm_message"]])
    text = await _listen(state, max_seconds=15.0)
    answer = detect_yes_no(text)

    if answer is False or answer is None:
        # try to use LLM to classify the response
        answer = await asyncio.to_thread(llm.detect_yes_no_llm_openai, text)

    if answer is False or answer is None:
        return phases.PHASE_HANGUP
    if answer is True:
        return phases.PHASE_END

    logger.info(
        "last_confirmation: could not classify response %r — hanging up",
        text,
    )
    return phases.PHASE_HANGUP
