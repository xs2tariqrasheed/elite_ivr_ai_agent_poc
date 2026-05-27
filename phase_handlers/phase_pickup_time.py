"""Phase 6b: collect the pickup time."""

import asyncio
import logging
from datetime import datetime

from fastapi import WebSocket

from constants import call_phases as phases
from services import llm
from services.duckling_service import parse_time_with_duckling

from phase_handlers.listen import _listen
from phase_handlers.speak import _speak


logger = logging.getLogger(__name__)


async def _run_phase_pickup_time(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PICKUP_TIME
    await _speak(websocket, state, [["pickup_time"]])
    text = (
        await _listen(
            state,
            max_seconds=15.0,
            initial_prompt=(
                "The caller is stating the pickup time for their limousine "
                "reservation. They may say a clock time like 9 AM or 3:30 PM, "
                "a time of day like morning, noon, afternoon, evening, or "
                "midnight, or a relative time like in an hour or right now."
            ),
            hotwords=(
                "AM, PM, a.m., p.m., o'clock, "
                "morning, noon, afternoon, evening, night, midnight, midday, "
                "early morning, late morning, early afternoon, late afternoon, "
                "early evening, late evening, tonight, "
                "quarter, half, past, to, sharp, "
                "one, two, three, four, five, six, seven, eight, nine, ten, "
                "eleven, twelve, thirteen, fourteen, fifteen, twenty, thirty, "
                "forty, fifty, "
                "hour, hours, minute, minutes, "
                "now, right now, in an hour, in half an hour, "
                "pickup, time"
            ),
        )
    ).strip()

    time_start = datetime.now()
    if text:
        transcript = text

        async def _apply_duckling_fallback() -> None:
            try:
                now = datetime.now()
                phrase = await asyncio.to_thread(
                    llm.normalize_time_for_duckling_openai, transcript, now
                )
                if not phrase:
                    logger.info(
                        "Duckling fallback: OpenAI returned no time phrase for %r",
                        transcript,
                    )
                    return
                duckling_time = await asyncio.to_thread(
                    parse_time_with_duckling, phrase, now
                )
                if duckling_time is not None:
                    state.reservation.pickup_time = duckling_time
                logger.info(
                    "Duckling pickup time: phrase=%r time=%s",
                    phrase,
                    state.reservation.pickup_time,
                )
            except Exception:
                logger.exception("Duckling fallback failed for pickup time")

        task = asyncio.create_task(_apply_duckling_fallback())
        state._background_tasks.add(task)
        task.add_done_callback(state._background_tasks.discard)

    time_end = datetime.now()
    logger.info(
        f"Time taken for _apply_duckling_fallback (time): {time_end - time_start} seconds"
    )

    return phases.PHASE_PICKUP_ADDRESS
