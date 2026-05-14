"""Helper: play an agent clip and wait for it to finish on the caller's line.

Imported by every phase handler that needs to vocalise a pre-recorded
prompt before listening for the caller's reply.
"""

import asyncio
import logging
from typing import Sequence

from fastapi import WebSocket

from services import agent_voice_service as voice


logger = logging.getLogger(__name__)


async def _speak(
    websocket: WebSocket,
    state,
    clips: Sequence[Sequence[str]],
    mark_timeout: float = 30.0,
) -> None:
    """Play one or more cached clips as a single utterance and wait for Twilio's mark echo.

    ``clips`` is a list of clip paths, where each clip path is itself a
    list of directory-walk segments relative to ``AUDIO_DIR``::

        await _speak(ws, state, [[audio_const.ACCOUNT_NAME]])
        await _speak(ws, state, [["account_names", "12345"], [audio_const.GREET_UNKNOWN]])

    All clips are concatenated into a single stream of frames and sent
    to Twilio as one unit followed by a single mark event, so this
    coroutine returns only after the entire sequence has played out on
    the caller's line.
    """
    state.agent_speaking = True
    state.capturing_audio = False
    state.reset_inbound_buffer()
    state.clear_mark()

    mark_name = await voice.play_audio(websocket, state.stream_sid, clips)
    received = await asyncio.to_thread(state.wait_for_mark, mark_timeout)
    if received != mark_name:
        logger.warning(
            "Mark echo timeout/mismatch (expected %s, got %r)", mark_name, received
        )

    state.agent_speaking = False
    state.clear_mark()
