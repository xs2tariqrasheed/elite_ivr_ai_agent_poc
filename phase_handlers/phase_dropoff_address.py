"""Phase 8: collect the drop-off address."""

import asyncio
import logging

from fastapi import WebSocket

from constants import call_phases as phases

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak
from services.text_to_speech_in_memory_service import text_to_speech_in_memory

logger = logging.getLogger(__name__)


async def _run_phase_dropoff_address(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_DROPOFF_ADDRESS
    await _speak(websocket, state, [["dropoff_address"]])
    dropoff_address = await _listen(state, max_seconds=15.0)
    state.reservation.dropoff_address = "JFK Airport, terminal 4"

    name = state.account_info.get("name", "John Smith")
    pickup_date = state.reservation.pickup_date
    pickup_time = state.reservation.pickup_time
    pickup_address = state.reservation.pickup_address
    dropoff_address = state.reservation.dropoff_address
    last_confirm_message = (
        f"[politely] Thanks. So here is what I have, Our sedan will pick up {name}, on... {pickup_date} at {pickup_time} "
        f"[politely] from... {pickup_address} and drop off at... {dropoff_address}."
        "[asking] Should I proceed and save this reservation? "
    )

    async def _prefetch_last_confirm_tts() -> None:
        try:
            await asyncio.to_thread(
                text_to_speech_in_memory,
                last_confirm_message,
                "last_confirm_message",
            )
        except Exception:
            logger.exception("Background TTS failed for last_confirm_message")

    task = asyncio.create_task(_prefetch_last_confirm_tts())
    state._background_tasks.add(task)
    task.add_done_callback(state._background_tasks.discard)

    return phases.PHASE_EMAIL_CONFIRMATION
