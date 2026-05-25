"""Phase 6b: collect the pickup time."""

import asyncio
import logging
from datetime import datetime

from fastapi import WebSocket

from constants import call_phases as phases
from services import llm

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


logger = logging.getLogger(__name__)


async def _run_phase_pickup_time(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PICKUP_TIME
    await _speak(websocket, state, [["pickup_time"]])
    text = (await _listen(state, max_seconds=15.0)).strip()

    time_start = datetime.now()
    if text:
        transcript = text

        async def _apply_openai_fallback() -> None:
            try:
                oa_time = await asyncio.to_thread(
                    llm.extract_pickup_time_openai, transcript
                )
                if oa_time is not None:
                    state.reservation.pickup_time = oa_time
                logger.info(
                    "OpenAI async fallback pickup time: time=%s",
                    state.reservation.pickup_time,
                )
            except Exception:
                logger.exception("OpenAI async fallback failed for pickup time")

        task = asyncio.create_task(_apply_openai_fallback())
        state._background_tasks.add(task)
        task.add_done_callback(state._background_tasks.discard)

    time_end = datetime.now()
    logger.info(
        f"Time taken for _apply_openai_fallback (time): {time_end - time_start} seconds"
    )

    return phases.PHASE_PICKUP_ADDRESS
