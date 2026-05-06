"""Helper: capture caller speech, run VAD, and transcribe via STT.

Imported by every phase handler that needs to wait for the caller to
respond after a prompt has been played.
"""

import asyncio
import logging
import time
from typing import Optional

from services import stt_service as stt
from utils.audio_utils import (
    detect_speech_boundaries,
    mulaw_to_float32_16k,
    reset_tail_last_five_stability_history,
)


logger = logging.getLogger(__name__)


async def _listen(
    state,
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
    max_seconds: float = 12.0,
    silence_after_speech_ms: int = 1200,
    grace_silence_ms: int = 5000,
    min_consecutive_speech_ms: int = 200,
    post_speech_timeout_ms: int = 15000,
) -> str:
    """Capture caller audio until VAD says they're done, then transcribe.

    ``grace_silence_ms`` is how long we'll wait for the caller to start
    speaking (RMS-energy VAD — see ``min_consecutive_speech_ms``) before
    giving up.

    ``post_speech_timeout_ms`` is a safety net: once speech has started, we
    never listen for more than this long total. If the VAD is thrown off by
    an unusually noisy line and speech_ended never triggers, this guarantees
    we still hand an audio buffer to STT and move the call forward.
    """

    # SENSITIVE CONFIG original was 1200
    # silence_after_speech_ms = 900

    # SENSITIVE CONFIG original was 12.0 see the argument max_seconds
    max_seconds = 200000.0

    state.reset_inbound_buffer()
    # The Twilio-glitch dedup window in audio_utils is module-global; clear
    # it at the start of every listen so state from the previous turn (or
    # from a previous call entirely) can't leak into this VAD evaluation.
    reset_tail_last_five_stability_history()
    state.capturing_audio = True
    start = time.time()
    speech_started = False
    speech_started_at_ms: Optional[float] = None
    last_buf_len = 0
    last_diag_ms = 0.0

    while True:
        # SENSITIVE CONFIG original was 0.2
        await asyncio.sleep(0.2)
        elapsed_ms = (time.time() - start) * 1000.0
        if elapsed_ms / 1000.0 >= max_seconds:
            logger.debug("listen: max duration hit (%.1fs)", max_seconds)
            break
        if not speech_started and elapsed_ms >= grace_silence_ms:
            # Caller never started speaking — bail out
            logger.debug(
                "listen: grace silence elapsed (%dms) without speech", grace_silence_ms
            )
            break
        if (
            speech_started
            and speech_started_at_ms is not None
            and (elapsed_ms - speech_started_at_ms) >= post_speech_timeout_ms
        ):
            logger.warning(
                "listen: post-speech timeout hit (%dms) — forcing end",
                post_speech_timeout_ms,
            )
            break

        buf_len = len(state.inbound_mulaw)
        # Diagnostic: log buffer growth every ~1 s so we can tell whether
        # the WS handler is actually streaming media into the buffer.
        if elapsed_ms - last_diag_ms >= 1000.0:
            logger.info(
                "listen-diag: elapsed=%.1fs buf_bytes=%d (+%d since last) "
                "capturing=%s speech_started=%s",
                elapsed_ms / 1000.0,
                buf_len,
                buf_len - last_buf_len,
                state.capturing_audio,
                speech_started,
            )
            last_diag_ms = elapsed_ms
        last_buf_len = buf_len

        if buf_len == 0:
            continue

        buf = bytes(state.inbound_mulaw)

        s_started, s_ended = detect_speech_boundaries(
            buf,
            silence_after_speech_ms=silence_after_speech_ms,
            min_consecutive_speech_ms=min_consecutive_speech_ms,
        )

        if s_started and not speech_started:
            speech_started = True
            speech_started_at_ms = elapsed_ms
        if s_ended:
            break

    state.capturing_audio = False
    audio_bytes = bytes(state.inbound_mulaw)
    state.reset_inbound_buffer()

    if not audio_bytes:
        logger.info("listen: no audio captured")
        return ""

    logger.debug("listen: captured %d mu-law bytes", len(audio_bytes))
    audio_np = mulaw_to_float32_16k(audio_bytes)
    start_time = time.time()
    text = await asyncio.to_thread(
        stt.transcribe_audio, audio_np, initial_prompt, hotwords
    )
    predict_time = time.time() - start_time
    logger.info(f"***** transcribe_audio: {predict_time} seconds")
    return text
