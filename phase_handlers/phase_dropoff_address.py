"""Phase 8: collect the drop-off address."""

import asyncio
import logging

from datetime import datetime

from fastapi import WebSocket

from constants import call_phases as phases

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak
from services.text_to_speech_in_memory_service import text_to_speech_in_memory

logger = logging.getLogger(__name__)


def _format_date_time(date: datetime or str) -> str:
    
    if isinstance(date, datetime):
        return date.strftime("%A %B %d")
    if isinstance(date, str):
        return datetime.strptime(date, "%Y-%m-%d")
    return ""

def _format_time(time: datetime or str) -> str:
    if isinstance(time, datetime):
        return time.strftime("%I:%M %p")
    if isinstance(time, str):
        return datetime.strptime(time, "%H:%M:%S").strftime("%I:%M %p")
    return ""

async def _run_phase_dropoff_address(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_DROPOFF_ADDRESS
    await _speak(websocket, state, [["dropoff_address"]])
    dropoff_address = await _listen(state, max_seconds=15.0)
    state.reservation.dropoff_address = "JFK Airport, terminal 4"

    name = state.account_info.get("name", "John Smith")
    pickup_date = state.reservation.pickup_date  or datetime.now().date()
    pickup_time = state.reservation.pickup_time or datetime.now().time()
    pickup_address = state.reservation.pickup_address
    dropoff_address = state.reservation.dropoff_address
    
    formatted_pickup_date = _format_date_time(pickup_date)
    formatted_pickup_time = _format_time(pickup_time)
    
    logger.info(f"formatted_pickup_date: {formatted_pickup_date}")
    logger.info(f"formatted_pickup_time: {formatted_pickup_time}")

    last_confirm_message = (
        f"[politely] Thanks. So here is what I have, Our sedan will pick up {name}, on... {formatted_pickup_date} at {formatted_pickup_time} "
        f"[politely] from... {pickup_address} and drop off at... {dropoff_address}."
        "[asking] Should I proceed and save this reservation? "
    )

    reservation_number_with_dashes = "-".join(
        list(state.reservation.reservation_number)
    )
    reservation_id_message = (
        f"[politely] Thanks. You are all set. Your confirmation number {reservation_number_with_dashes} will be mailed to your email address. "
        "[politely] Thank you for calling Elite Limousine. Goodbye."
    )

    print(f"reservation_id_message: {reservation_id_message}")
    
    print(f"last_confirm_message: {last_confirm_message}")

    async def _prefetch_last_confirm_tts() -> None:
        try:
            await asyncio.to_thread(
                text_to_speech_in_memory,
                last_confirm_message,
                "last_confirm_message",
            )
            await asyncio.to_thread(
                text_to_speech_in_memory,
                reservation_id_message,
                "reservation_id_message",
            )
        except Exception:
            logger.exception("Background TTS failed for last_confirm_message")

    task = asyncio.create_task(_prefetch_last_confirm_tts())
    state._background_tasks.add(task)
    task.add_done_callback(state._background_tasks.discard)

    return phases.PHASE_EMAIL_CONFIRMATION
