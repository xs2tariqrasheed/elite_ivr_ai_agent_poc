"""Adapts Twilio Media Streams to the WebSocket interface the pipeline expects.

Twilio sends and receives base64 G.711 μ-law at 8 kHz. Outbound TTS is emitted
as μ-law so it plays back natively (see configs.TWILIO_AUDIO), but inbound caller
audio is transcoded to PCM16@16k (see services.audio) before STT, since
AssemblyAI's universal-streaming model only transcribes reliably at 16 kHz. The
pipeline talks to this object exactly as it talks to a browser WebSocket; only
the wire framing differs, so AudioBridge and TurnHandler need no Twilio
awareness.
"""
import base64
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from services.audio import mulaw8k_to_pcm16_16k

log = logging.getLogger("twilio")


class TwilioTransport:
    """Wraps a Twilio Media Stream WebSocket, exposing a browser-WebSocket shape."""

    # AssemblyAI v3 requires 50–1000 ms of audio per message; Twilio streams
    # 20 ms frames (160 μ-law bytes @ 8 kHz). Coalesce frames to ~100 ms before
    # forwarding so STT doesn't reject the stream (close code 3007).
    _CHUNK_BYTES = 800

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._stream_sid: str | None = None
        # custom <Parameter> values declared in the TwiML <Stream>, e.g. agent.
        self.custom_parameters: dict = {}
        self._inbound = bytearray()
        self._forwarded = False

    async def accept(self) -> None:
        await self._ws.accept()

    async def start(self) -> None:
        """Consume Twilio's `connected`/`start` frames and capture the stream SID.

        Called once before the pipeline runs so the SID is known before any
        outbound media (e.g. the agent's opening greeting) is sent.
        """
        while self._stream_sid is None:
            try:
                evt = json.loads(await self._ws.receive_text())
            except (WebSocketDisconnect, RuntimeError):
                return
            except (ValueError, TypeError):
                continue
            if evt.get("event") == "start":
                self._capture_start(evt)

    def _capture_start(self, evt: dict) -> None:
        start = evt.get("start") or {}
        self._stream_sid = start.get("streamSid")
        self.custom_parameters = start.get("customParameters") or {}
        log.info("Twilio stream started: %s", self._stream_sid)

    async def receive(self) -> dict:
        """Translate one Twilio frame into AudioBridge's {bytes|text|type} shape."""
        try:
            raw = await self._ws.receive_text()
        except (WebSocketDisconnect, RuntimeError):
            return {"type": "websocket.disconnect"}
        try:
            evt = json.loads(raw)
        except (ValueError, TypeError):
            return {}

        event = evt.get("event")
        if event == "media":
            self._inbound.extend(base64.b64decode(evt["media"]["payload"]))
            if len(self._inbound) >= self._CHUNK_BYTES:
                # Coalesced μ-law@8k -> PCM16@16k for AssemblyAI (see services.audio).
                chunk = mulaw8k_to_pcm16_16k(bytes(self._inbound))
                self._inbound.clear()
                if not self._forwarded:
                    self._forwarded = True
                    log.info("Twilio inbound audio forwarding to STT")
                return {"bytes": chunk}
            return {}
        if event == "stop":
            return {"type": "websocket.disconnect"}
        if event == "start" and self._stream_sid is None:
            self._capture_start(evt)
        # connected / mark / dtmf and anything else: nothing for the pipeline.
        return {}

    async def send_bytes(self, audio: bytes) -> None:
        """Send a μ-law TTS chunk back to the caller as a Twilio media frame."""
        if self._stream_sid is None:
            return
        await self._ws.send_text(json.dumps({
            "event": "media",
            "streamSid": self._stream_sid,
            "media": {"payload": base64.b64encode(audio).decode("ascii")},
        }))

    async def send_json(self, msg: dict) -> None:
        """Drop pipeline control/UI messages that have no Twilio equivalent.

        The browser-only UI messages (partial/final/agent/state/timings/...) are
        meaningless on a phone call, and the pipeline is half-duplex with no
        barge-in, so there is no `interrupt`/`clear` flush to forward either.
        """
        return

    async def close(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass
