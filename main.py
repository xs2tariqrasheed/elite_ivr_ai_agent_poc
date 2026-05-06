"""FastAPI entry point for the Elite Limousine IVR agent.

Endpoints:
    POST /voice          - Twilio webhook; returns TwiML that opens a Media
                           Stream WebSocket back to this server.
    WS   /ws/twilio      - Twilio Media Stream; receives caller audio and
                           plays back the agent's pre-recorded responses.

The whole call flow is non-interruptible: the agent always finishes the
current clip before we start listening for the caller's next utterance.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.rest import Client as TwilioClient

import config
from logging_config import setup_logging
from constants import audio_files as audio_const
from constants import call_phases as phases
from services import account_service
from services import agent_voice_service as voice
from services import call_state_service
from services import intent_service as intent
from services import llm_service as llm
from services import phase_manager
from services import stt_service as stt
from services.number_extractor import extract_number
from services.phone_extractor import extract_phone
from utils.audio_utils import (
    detect_speech_boundaries,
    mulaw_to_float32_16k,
    reset_tail_last_five_stability_history,
)
from utils.misc import is_valid_account_number, is_valid_phone


setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Elite Limousine IVR")


# ---------------------------------------------------------------------------
# Startup: pre-load every model and audio file
# ---------------------------------------------------------------------------


@app.on_event("startup")
def on_startup() -> None:
    logger.info("Booting Elite IVR …")
    voice.load_audio_files()
    stt.load_model()
    intent.load_model()
    account_service.load_accounts()
    llm.warm_up_model()
    logger.info("Boot complete")


# ---------------------------------------------------------------------------
# /voice — Twilio webhook
# ---------------------------------------------------------------------------


@app.post("/voice")
async def voice_webhook(request: Request) -> Response:
    """Return TwiML that asks Twilio to open a Media Stream to us."""
    ws_url = config.TWILIO_STREAM_WS_URL
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{ws_url}" />'
        "</Connect>"
        "</Response>"
    )
    logger.info("Returning TwiML pointing Twilio at %s", ws_url)
    return Response(content=twiml, media_type="application/xml")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


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
# Phase implementations
# ---------------------------------------------------------------------------


async def _run_phase_intent(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_INTENT
    await _speak(websocket, state, audio_const.GREET_UNKNOWN)
    text = await _listen(
        state,
        initial_prompt="The caller is calling Elite Limousine to make a reservation or ask for help.",
        hotwords=(
            "reservation, booking, car, limousine, limo, vehicle, "
            "ride, pickup, dropoff, chxzauffeur, airport, sedan, "
            "SUV, luxury, transportation, availability, cancel, "
            "modify, reschedule, tonight, tomorrow"
        ),
    )

    start_time = time.time()
    label = intent.classify_with_threshold(text)
    predict_time = time.time() - start_time
    logger.info(f"***** classify_with_threshold: {predict_time}")

    if label == intent.INTENT_NEW_RESERVATION:
        return phases.PHASE_ACCOUNT_NUMBER

    # "other" intent — apologise and hang up
    await _speak(websocket, state, audio_const.OTHER_INTENT)
    return phases.PHASE_HANGUP


async def _run_phase_account_number(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_ACCOUNT_NUMBER

    for attempt in (1, 2):  # one initial try, one retry
        clip = (
            audio_const.ACCOUNT_NUMBER
            if attempt == 1
            else audio_const.ACCOUNT_NUMBER_RETRY
        )
        await _speak(websocket, state, clip)
        text = await _listen(
            state,
            initial_prompt="The caller will say their 4-digit account number.",
            hotwords="account number digits",
        )
        start_time = time.time()
        account_number = extract_number(text)
        if not is_valid_account_number(account_number):
            account_number = await asyncio.to_thread(llm.extract_account_number, text)
        predict_time = time.time() - start_time
        logger.info(
            f"***** extract_account_number: {predict_time} seconds, account_number: {account_number}, is_with_llm: {not is_valid_account_number(account_number)}"
        )

        state.increment_attempts("account_number")

        if account_number is None:
            logger.info(
                "account_number: extraction failed on attempt %d (text=%r)",
                attempt,
                text,
            )
            continue

        record = account_service.find_account_by_number(account_number)
        if record is None:
            logger.info("account_number: %s not found in dummy data", account_number)
            await _speak(websocket, state, audio_const.ACCOUNT_NOT_FOUND)
            return phases.PHASE_HANGUP

        # Success
        state.reservation.account_number = account_number
        state.reservation.account_record = record
        return phases.PHASE_ACCOUNT_NAME

    # Two failed attempts
    logger.info("account_number: exhausted retries — hanging up")
    await _speak(websocket, state, audio_const.ACCOUNT_NOT_FOUND)
    return phases.PHASE_HANGUP


async def _run_phase_account_name(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_ACCOUNT_NAME
    await _speak(websocket, state, audio_const.ACCOUNT_NAME)
    state.reservation.account_name = (await _listen(state)).strip()
    return phases.PHASE_FIRST_NAME


async def _run_phase_first_name(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_FIRST_NAME
    await _speak(websocket, state, audio_const.FIRST_NAME)
    state.reservation.first_name = (await _listen(state)).strip()
    return phases.PHASE_LAST_NAME


async def _run_phase_last_name(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_LAST_NAME
    await _speak(websocket, state, audio_const.LAST_NAME)
    state.reservation.last_name = (await _listen(state)).strip()
    return phases.PHASE_PICKUP_DATE_TIME


async def _run_phase_pickup_date_time(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PICKUP_DATE_TIME
    await _speak(websocket, state, audio_const.PICKUP_DATE_TIME)
    state.reservation.pickup_date_time = (
        await _listen(state, max_seconds=15.0)
    ).strip()
    return phases.PHASE_PICKUP_ADDRESS


async def _run_phase_pickup_address(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_PICKUP_ADDRESS
    await _speak(websocket, state, audio_const.PICKUP_ADDRESS)
    state.reservation.pickup_address = (await _listen(state, max_seconds=15.0)).strip()
    return phases.PHASE_DROPOFF_ADDRESS


async def _run_phase_dropoff_address(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_DROPOFF_ADDRESS
    await _speak(websocket, state, audio_const.DROPOFF_ADDRESS)
    state.reservation.dropoff_address = (await _listen(state, max_seconds=15.0)).strip()
    return phases.PHASE_CALLBACK_NUMBER


async def _run_phase_callback_number(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_CALLBACK_NUMBER
    await _speak(websocket, state, audio_const.CALLBACK_NUMBER)
    text = await _listen(
        state,
        initial_prompt="The caller will say a 10-digit US phone number.",
        hotwords="phone number digits",
    )
    start_time = time.time()
    phone = extract_phone(text)
    if not is_valid_phone(phone):
        phone = await asyncio.to_thread(llm.extract_phone_number, text)
    predict_time = time.time() - start_time
    logger.info(
        f"***** extract_phone: {predict_time} seconds, phone: {phone}, is_with_llm: {not is_valid_phone(phone)}"
    )
    state.reservation.callback_number = phone if phone else text.strip()
    return phases.PHASE_EMAIL


async def _run_phase_email(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_EMAIL
    await _speak(websocket, state, audio_const.EMAIL)
    state.reservation.email = (await _listen(state, max_seconds=15.0)).strip()
    return phases.PHASE_END


async def _run_phase_end(websocket: WebSocket, state) -> str:
    state.phase = phases.PHASE_END
    logger.info(
        "Reservation summary for call %s: %s",
        state.call_sid,
        state.reservation.as_summary_dict(),
    )
    await _speak(websocket, state, audio_const.GOOD_BYE)
    return phases.PHASE_HANGUP


_PHASE_HANDLERS = {
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


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/twilio")
async def twilio_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Twilio WebSocket connected")

    state = None
    flow_task: Optional[asyncio.Task] = None

    # Diagnostic counters for WS media events.
    media_total = 0
    media_captured = 0
    media_dropped_no_state = 0
    media_dropped_not_capturing = 0
    last_media_log_ms = time.time() * 1000.0

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                logger.exception("Bad JSON from Twilio")
                continue

            event = msg.get("event")

            if event == "connected":
                logger.info("Twilio reports connected: %s", msg)

            elif event == "start":
                start = msg.get("start", {})
                stream_sid = start.get("streamSid") or msg.get("streamSid")
                call_sid = start.get("callSid")
                logger.info(
                    "Stream start: stream_sid=%s call_sid=%s", stream_sid, call_sid
                )
                state = call_state_service.create_call_state(call_sid, stream_sid)
                flow_task = asyncio.create_task(_run_call_flow(websocket, state))

            elif event == "media":
                media_total += 1
                if state is None:
                    media_dropped_no_state += 1
                    continue
                if not state.capturing_audio:
                    media_dropped_not_capturing += 1
                    continue
                payload_b64 = msg.get("media", {}).get("payload", "")
                if not payload_b64:
                    continue
                try:
                    chunk = base64.b64decode(payload_b64)
                    state.inbound_mulaw.extend(chunk)
                    media_captured += 1
                except Exception:
                    logger.exception("Failed to decode media payload")

                now_ms = time.time() * 1000.0
                if now_ms - last_media_log_ms >= 1000.0:
                    logger.info(
                        "ws-media-diag: total=%d captured=%d dropped_no_state=%d "
                        "dropped_not_capturing=%d capturing=%s",
                        media_total,
                        media_captured,
                        media_dropped_no_state,
                        media_dropped_not_capturing,
                        state.capturing_audio if state is not None else "n/a",
                    )
                    last_media_log_ms = now_ms

            elif event == "mark":
                mark_name = msg.get("mark", {}).get("name", "")
                logger.debug("Twilio mark echoed: %s", mark_name)
                if state is not None:
                    state.signal_mark(mark_name)

            elif event == "stop":
                # remove call state
                call_state_service.remove_call_state(state.stream_sid)
                logger.info("Twilio reports stream stop")
                break

            else:
                call_state_service.remove_call_state(state.stream_sid)
                logger.debug("Unhandled Twilio event: %s", event)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket loop crashed")
    finally:
        if flow_task and not flow_task.done():
            flow_task.cancel()
            try:
                await flow_task
            except (asyncio.CancelledError, Exception):
                pass
        if state is not None:
            call_state_service.remove_call_state(state.stream_sid)
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Manual entry point: ``python main.py``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower(),
    )
