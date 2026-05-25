"""Phase 6a: collect the pickup date."""

import asyncio
import logging
from datetime import datetime

from fastapi import WebSocket

from constants import call_phases as phases
from services import llm

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


logger = logging.getLogger(__name__)


async def _run_phase_pickup_date(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PICKUP_DATE
    await _speak(websocket, state, [["pickup_date"]])
    text = (await _listen(state, max_seconds=15.0)).strip()

    time_start = datetime.now()
    if text:
        transcript = text

        async def _apply_openai_fallback() -> None:
            try:
                oa_date = await asyncio.to_thread(
                    llm.extract_pickup_date_openai, transcript
                )
                if oa_date is not None:
                    state.reservation.pickup_date = oa_date
                logger.info(
                    "OpenAI async fallback pickup date: date=%s",
                    state.reservation.pickup_date,
                )
            except Exception:
                logger.exception("OpenAI async fallback failed for pickup date")

        task = asyncio.create_task(_apply_openai_fallback())
        state._background_tasks.add(task)
        task.add_done_callback(state._background_tasks.discard)

    time_end = datetime.now()
    logger.info(
        f"Time taken for _apply_openai_fallback (date): {time_end - time_start} seconds"
    )

    return phases.PHASE_PICKUP_TIME
