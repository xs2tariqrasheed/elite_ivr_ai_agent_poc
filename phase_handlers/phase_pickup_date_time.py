"""Phase 6: collect the pickup date and time."""

import asyncio
import logging
from datetime import datetime
from fastapi import WebSocket

from constants import call_phases as phases
from services import llm

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


logger = logging.getLogger(__name__)


async def _run_phase_pickup_date_time(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PICKUP_DATE_TIME
    await _speak(websocket, state, [["pickup_date_time"]])
    text = (await _listen(state, max_seconds=15.0)).strip()

    # today = datetime.now()

    # time_start = datetime.now()
    # date, time = await asyncio.to_thread(llm.extract_pickup_date_time, text, today)
    # time_end = datetime.now()
    # logger.info(
    #     f"Time taken for extract_pickup_date_time: {time_end - time_start} seconds"
    # )

    # state.reservation.pickup_date = date
    # state.reservation.pickup_time = time

    # logger.info(
    #     "Captured pickup date/time: date=%s time=%s (from %r)",
    #     date,
    #     time,
    #     text,
    # )

    time_start = datetime.now()
    date = None
    if date is None and text:
        transcript = text

        async def _apply_openai_fallback() -> None:
            try:
                oa_date, oa_time = await asyncio.to_thread(
                    llm.extract_pickup_date_time_openai, transcript
                )
                if oa_date is not None:
                    state.reservation.pickup_date = oa_date
                if state.reservation.pickup_time is None and oa_time is not None:
                    state.reservation.pickup_time = oa_time
                logger.info(
                    "OpenAI async fallback pickup date/time: date=%s time=%s",
                    state.reservation.pickup_date,
                    state.reservation.pickup_time,
                )
            except Exception:
                logger.exception("OpenAI async fallback failed for pickup date/time")

        task = asyncio.create_task(_apply_openai_fallback())
        state._background_tasks.add(task)
        task.add_done_callback(state._background_tasks.discard)

    time_end = datetime.now()
    logger.info(
        f"Time taken for _apply_openai_fallback: {time_end - time_start} seconds"
    )

    return phases.PHASE_PICKUP_ADDRESS
