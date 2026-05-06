"""Phase 2: ask for and validate the caller's 4-digit account number."""

import asyncio
import logging
import time

from fastapi import WebSocket

from constants import audio_files as audio_const
from constants import call_phases as phases
from services import account_service
from services import llm_service as llm
from services.number_extractor import extract_number
from utils.misc import is_valid_account_number

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


logger = logging.getLogger(__name__)


async def _run_phase_account_number(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_ACCOUNT_NUMBER

    for attempt in (1, 2):  # one initial try, one retry
        clip = (
            audio_const.ACCOUNT_NUMBER
            if attempt == 1
            else audio_const.ACCOUNT_NUMBER_RETRY
        )
        await _speak(websocket, state, clip)
        text = await _listen(
            state,
            initial_prompt="The caller will say their 4-digit account number.",
            hotwords="account number digits",
        )
        start_time = time.time()
        account_number = extract_number(text)
        if not is_valid_account_number(account_number):
            account_number = await asyncio.to_thread(llm.extract_account_number, text)
        predict_time = time.time() - start_time
        logger.info(
            f"***** extract_account_number: {predict_time} seconds, account_number: {account_number}, is_with_llm: {not is_valid_account_number(account_number)}"
        )

        state.increment_attempts("account_number")

        if account_number is None:
            logger.info(
                "account_number: extraction failed on attempt %d (text=%r)",
                attempt,
                text,
            )
            continue

        record = account_service.find_account_by_number(account_number)
        if record is None:
            logger.info("account_number: %s not found in dummy data", account_number)
            await _speak(websocket, state, audio_const.ACCOUNT_NOT_FOUND)
            return phases.PHASE_HANGUP

        # Success
        state.reservation.account_number = account_number
        state.reservation.account_record = record
        return phases.PHASE_ACCOUNT_NAME

    # Two failed attempts
    logger.info("account_number: exhausted retries — hanging up")
    await _speak(websocket, state, audio_const.ACCOUNT_NOT_FOUND)
    return phases.PHASE_HANGUP
