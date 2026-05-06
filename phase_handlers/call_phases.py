"""Call flow orchestration.

This module owns the cross-phase concerns of a Twilio call:

* The shared ``_speak`` / ``_listen`` helpers used by every phase handler.
* ``_hangup_call`` — Twilio REST hang-up for cleanly ending a call.
* ``_PHASE_HANDLERS`` — phase identifier → handler coroutine dispatch.
* ``_run_call_flow`` — drives the call from PHASE_INTENT through hang-up.

Each individual phase handler lives in its own sibling module within the
``phase_handlers`` package and imports ``_speak`` / ``_listen`` from
here. To avoid a circular import (``call_phases`` → phase modules →
``call_phases``), the dispatch table is built lazily inside
``_run_call_flow`` rather than at module import time.
"""

import asyncio
import logging
import time
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from twilio.rest import Client as TwilioClient

import config
from constants import call_phases as phases
from services import agent_voice_service as voice
from services import call_state_service
from services import phase_manager
from services import stt_service as stt
from utils.audio_utils import (
    detect_speech_boundaries,
    mulaw_to_float32_16k,
    reset_tail_last_five_stability_history,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Twilio call hang-up via REST
# ---------------------------------------------------------------------------


def _hangup_call(call_sid: str) -> None:
    if not (config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN):
        logger.warning("Twilio creds missing — cannot hang up call %s", call_sid)
        return
    try:
        client = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        client.calls(call_sid).update(status="completed")
        call_state_service.remove_call_state(call_sid)
        logger.info("Hung up call %s via Twilio REST", call_sid)
    except Exception:
        logger.exception("Failed to hang up call %s", call_sid)


# ---------------------------------------------------------------------------
# Helper: play an agent clip and wait for it to finish on the caller's line
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helper: capture caller speech and run STT
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase dispatch table (built lazily to avoid circular imports)
# ---------------------------------------------------------------------------


_PHASE_HANDLERS: Optional[dict] = None


def _build_phase_handlers() -> dict:
    """Import every phase handler and assemble the dispatch table.

    Imported lazily so that the individual phase modules can themselves
    import ``_speak`` / ``_listen`` from this module without creating a
    circular import at package load time.
    """
    from phase_handlers.phase_intent import _run_phase_intent
    from phase_handlers.phase_account_number import _run_phase_account_number
    from phase_handlers.phase_account_name import _run_phase_account_name
    from phase_handlers.phase_first_name import _run_phase_first_name
    from phase_handlers.phase_last_name import _run_phase_last_name
    from phase_handlers.phase_pickup_date_time import _run_phase_pickup_date_time
    from phase_handlers.phase_pickup_address import _run_phase_pickup_address
    from phase_handlers.phase_dropoff_address import _run_phase_dropoff_address
    from phase_handlers.phase_callback_number import _run_phase_callback_number
    from phase_handlers.phase_email import _run_phase_email
    from phase_handlers.phase_end import _run_phase_end

    return {
        phases.PHASE_INTENT: _run_phase_intent,
        phases.PHASE_ACCOUNT_NUMBER: _run_phase_account_number,
        phases.PHASE_ACCOUNT_NAME: _run_phase_account_name,
        phases.PHASE_FIRST_NAME: _run_phase_first_name,
        phases.PHASE_LAST_NAME: _run_phase_last_name,
        phases.PHASE_PICKUP_DATE_TIME: _run_phase_pickup_date_time,
        phases.PHASE_PICKUP_ADDRESS: _run_phase_pickup_address,
        phases.PHASE_DROPOFF_ADDRESS: _run_phase_dropoff_address,
        phases.PHASE_CALLBACK_NUMBER: _run_phase_callback_number,
        phases.PHASE_EMAIL: _run_phase_email,
        phases.PHASE_END: _run_phase_end,
    }


# ---------------------------------------------------------------------------
# The orchestrator coroutine
# ---------------------------------------------------------------------------


async def _run_call_flow(websocket: WebSocket, state) -> None:
    """Drive the call from PHASE_INTENT through to hang-up."""
    global _PHASE_HANDLERS
    if _PHASE_HANDLERS is None:
        _PHASE_HANDLERS = _build_phase_handlers()

    current = phases.PHASE_INTENT
    try:
        while not phase_manager.is_terminal(current):
            handler = _PHASE_HANDLERS.get(current)
            if handler is None:
                logger.error("No handler for phase %r — aborting", current)
                break
            logger.info("Entering %s", current)
            current = await handler(websocket, state)
            logger.info("Phase returned %s", current)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected mid-flow")
        return
    except Exception:
        logger.exception("Call flow crashed")
    finally:
        # Always try to hang up cleanly when we exit the loop.
        if state.call_sid:
            await asyncio.to_thread(_hangup_call, state.call_sid)
        try:
            await voice.send_clear(websocket, state.stream_sid)
        except Exception:
            pass
