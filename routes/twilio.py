"""Twilio webhook and Media Stream WebSocket routes."""
import asyncio
import base64
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

import config
from services import call_state_service
from phase_handlers.call_phase_registry import _run_call_flow

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/voice")
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


@router.websocket("/ws/twilio")
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
