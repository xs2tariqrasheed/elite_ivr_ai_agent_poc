"""Twilio Voice integration.

  POST/GET /twilio/voice   — TwiML webhook; connects the call to the media stream
  WS       /twilio/stream  — Twilio Media Stream; one VoiceSession per call

Point a Twilio phone number's "A call comes in" webhook at /twilio/voice. The
returned TwiML tells Twilio to open a bidirectional Media Stream to the wss://
URL in TWILIO_STREAM_WS_URL, which should resolve to /twilio/stream.
"""
import logging
from urllib.parse import parse_qs
from xml.sax.saxutils import quoteattr

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import Response

from configs.settings import TWILIO_AUDIO, settings
from db.accounts import get_account_by_phone
from services.twilio_transport import TwilioTransport
from services.voice_session import VoiceSession

log = logging.getLogger("twilio")

router = APIRouter()


@router.api_route("/twilio/voice", methods=["GET", "POST"])
async def twilio_voice(request: Request):
    """Return TwiML that bridges the inbound call to our Media Stream socket.

    Twilio sends the caller's number as `From`; we forward it as a custom
    <Parameter> so the Media Stream's `start` frame carries it through to the
    agent (the start frame itself doesn't include the caller ID).
    """
    if request.method == "POST":
        # Twilio posts application/x-www-form-urlencoded; parse without the
        # python-multipart dependency that request.form() would require.
        body = (await request.body()).decode("utf-8", "replace")
        caller = (parse_qs(body).get("From") or [""])[0]
    else:
        caller = request.query_params.get("From") or ""
    stream_url = settings.twilio_stream_ws_url
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f"<Stream url={quoteattr(stream_url)}>"
        f"<Parameter name=\"from\" value={quoteattr(caller)} />"
        "</Stream>"
        "</Connect></Response>"
    )
    return Response(content=twiml, media_type="text/xml")


@router.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket):
    log.info("Twilio /twilio/stream WS connection opened")
    transport = TwilioTransport(ws)
    await transport.accept()
    # Block until Twilio's `start` frame so the stream SID (needed to send audio
    # back) is known before the opening greeting is synthesized.
    await transport.start()

    agent_name = transport.custom_parameters.get("agent") or settings.agent

    params: dict = {}
    caller = transport.custom_parameters.get("from")
    if caller:
        account = get_account_by_phone(caller)
        if account is not None:
            params["account"] = account
            log.info("Twilio caller %s matched account %s", caller, account["name"])
        else:
            log.info("Twilio caller %s did not match any account", caller)

    log.info("Twilio call connected; agent=%s", agent_name)
    await VoiceSession(
        transport, settings, agent_name=agent_name, params=params,
        audio_format=TWILIO_AUDIO,
    ).run()
