"""Adapts a raw browser WebSocket to the transport interface the pipeline uses.

The pipeline talks to either this wrapper (browser) or `TwilioTransport` (phone)
through the same small surface — receive / send_bytes / send_json / clear /
close — so AudioBridge and TurnHandler need no transport awareness. The only
reason this wrapper exists (rather than passing the raw `WebSocket`) is `clear()`:
the barge-in flush must exist on every transport, and a raw FastAPI WebSocket has
no such method. Everything else is a thin pass-through.
"""
from fastapi import WebSocket


class BrowserTransport:
    """Wraps a browser WebSocket, exposing the same shape as TwilioTransport."""

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws

    async def accept(self) -> None:
        await self._ws.accept()

    async def receive(self) -> dict:
        return await self._ws.receive()

    async def send_bytes(self, audio: bytes) -> None:
        await self._ws.send_bytes(audio)

    async def send_json(self, msg: dict) -> None:
        await self._ws.send_json(msg)

    async def clear(self) -> None:
        """Tell the browser client to drop its queued/playing audio (barge-in).

        The out-of-repo browser client listens for this control message and
        flushes its WebAudio playback queue so the agent stops instantly when the
        caller interrupts — the browser equivalent of Twilio's `clear` event.
        """
        await self._ws.send_json({"type": "interrupt"})

    async def close(self) -> None:
        try:
            await self._ws.close()
        except Exception:
            pass
