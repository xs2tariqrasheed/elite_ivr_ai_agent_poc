"""Phase 9: collect the caller's 10-digit callback phone number."""

import asyncio
import logging
import time

from fastapi import WebSocket

from constants import audio_files as audio_const
from constants import call_phases as phases
from services import llm
from services.phone_extractor import extract_phone
from utils.misc import is_valid_phone

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


logger = logging.getLogger(__name__)


async def _run_phase_callback_number(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_CALLBACK_NUMBER
    await _speak(websocket, state, [[audio_const.CALLBACK_NUMBER]])
    text = await _listen(
        state,
        initial_prompt="The caller will say a 10-digit US phone number.",
        hotwords="phone number digits",
    )
    start_time = time.time()
    phone = extract_phone(text)
    if not is_valid_phone(phone):
        phone = await asyncio.to_thread(llm.extract_phone_number, text)
    predict_time = time.time() - start_time
    logger.info(
        f"***** extract_phone: {predict_time} seconds, phone: {phone}, is_with_llm: {not is_valid_phone(phone)}"
    )
    state.reservation.callback_number = phone if phone else text.strip()
    return phases.PHASE_EMAIL
