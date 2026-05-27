"""Phase 6a: collect the pickup date."""

import asyncio
import logging
from datetime import datetime

from fastapi import WebSocket

from constants import call_phases as phases
from services import llm
from services.duckling_service import parse_date_with_duckling

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

        async def _apply_duckling_fallback() -> None:
            try:
                now = datetime.now()
                phrase = await asyncio.to_thread(
                    llm.normalize_for_duckling_openai, transcript, now
                )
                if not phrase:
                    logger.info(
                        "Duckling fallback: OpenAI returned no phrase for %r",
                        transcript,
                    )
                    return
                duckling_date = await asyncio.to_thread(
                    parse_date_with_duckling, phrase, now
                )
                if duckling_date is not None:
                    state.reservation.pickup_date = duckling_date
                logger.info(
                    "Duckling pickup date: phrase=%r date=%s",
                    phrase,
                    state.reservation.pickup_date,
                )
            except Exception:
                logger.exception("Duckling fallback failed for pickup date")

        task = asyncio.create_task(_apply_duckling_fallback())
        state._background_tasks.add(task)
        task.add_done_callback(state._background_tasks.discard)

    time_end = datetime.now()
    logger.info(
        f"Time taken for _apply_duckling_fallback (date): {time_end - time_start} seconds"
    )

    return phases.PHASE_PICKUP_TIME
