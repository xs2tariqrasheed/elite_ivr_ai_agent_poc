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
    text = (
        await _listen(
            state,
            max_seconds=15.0,
            initial_prompt=(
                "The caller is stating the pickup date for their limousine "
                "reservation. They may say a weekday, a calendar date, a "
                "holiday, or a relative day like today, tomorrow, or next week."
            ),
            hotwords=(
                "today, tomorrow, tonight, day after tomorrow, this, next, "
                "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, "
                "January, February, March, April, May, June, July, August, "
                "September, October, November, December, "
                "first, second, third, fourth, fifth, sixth, seventh, eighth, "
                "ninth, tenth, eleventh, twelfth, thirteenth, fourteenth, "
                "fifteenth, sixteenth, seventeenth, eighteenth, nineteenth, "
                "twentieth, thirtieth, "
                "weekend, weekday, morning, evening, "
                "Christmas, Christmas Eve, New Year, New Year's Eve, "
                "Thanksgiving, Easter, Memorial Day, Labor Day, Independence Day, "
                "pickup, reservation, date"
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
