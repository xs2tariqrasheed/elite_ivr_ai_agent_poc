"""Helper: play an agent clip and wait for it to finish on the caller's line.

Imported by every phase handler that needs to vocalise a pre-recorded
prompt before listening for the caller's reply.
"""

import asyncio
import logging

from fastapi import WebSocket

from services import agent_voice_service as voice


logger = logging.getLogger(__name__)


async def _speak(
    websocket: WebSocket, state, audio_name: str, mark_timeout: float = 30.0
) -> None:
    """Play ``audio_name`` and wait for Twilio to echo back the mark."""
    state.agent_speaking = True
    state.capturing_audio = False
    state.reset_inbound_buffer()
    state.clear_mark()

    mark_name = await voice.play_audio(websocket, state.stream_sid, audio_name)
    received = await asyncio.to_thread(state.wait_for_mark, mark_timeout)
    if received != mark_name:
        logger.warning(
            "Mark echo timeout/mismatch (expected %s, got %r)", mark_name, received
        )

    state.agent_speaking = False
    state.clear_mark()
