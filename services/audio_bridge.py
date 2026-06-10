"""Routes audio between the browser, AssemblyAI STT, and the agent."""
import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from services.pipeline_state import PipelineState
from services.stt import AssemblyAIStream

log = logging.getLogger("voice")

# Extra seconds to keep muting caller audio after the agent's playback is
# projected to end, covering Twilio's jitter buffer and echo tail. Too small
# and the agent's voice bleeds into STT; too large clips the caller's first
# word. Tune against real calls.
_ECHO_TAIL_SECONDS = 0.7

# Mean absolute PCM16 amplitude above which an inbound chunk is treated as
# caller speech (vs. line silence / comfort noise) — used only for diagnostic
# voice-activity logging, independent of AssemblyAI. Tune against real calls.
_VOICE_LEVEL = 400


class AudioBridge:
    """Bridges browser mic audio through STT and dispatches final turns to the agent."""

    def __init__(
        self,
        client: WebSocket,
        stt: AssemblyAIStream,
        state: PipelineState,
        on_turn: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        self._client = client
        self._stt = stt
        self._state = state
        self._on_turn = on_turn  # handle_turn(text, user_stopped_at=..., gap_filler=...)
        # Turns are queued and run one at a time. The pipeline is half-duplex and
        # never interrupts the bot, so overlapping turns would otherwise run
        # concurrently — sharing the agent's memory thread and the single
        # outbound audio stream — and the replies would interleave out of sync.
        self._turns: asyncio.Queue[tuple[str, float | None, bool]] = asyncio.Queue()

    def enqueue_turn(
        self,
        text: str,
        user_stopped_at: float | None = None,
        gap_filler: bool = False,
    ) -> None:
        self._turns.put_nowait((text, user_stopped_at, gap_filler))

    async def turn_worker(self) -> None:
        """Run queued turns strictly in order, one fully finishing before the next."""
        while True:
            text, user_stopped_at, gap_filler = await self._turns.get()
            log.info("Turn worker start: %r", text)
            task = asyncio.create_task(
                self._on_turn(
                    text, user_stopped_at=user_stopped_at, gap_filler=gap_filler
                )
            )
            self._state.turn_task = task
            try:
                await task
            except asyncio.CancelledError:
                task.cancel()
                raise
            log.info("Turn worker done: %r", text)

    async def browser_to_stt(self) -> None:
        """Forward mic PCM frames to AssemblyAI, muted while the agent speaks.

        Half-duplex: with barge-in disabled, feeding caller audio to STT while
        the agent is audible lets the agent's own voice (line/acoustic echo on a
        phone call) bleed back in — corrupting transcripts and stalling
        end-of-turn detection. While the agent is speaking we send equal-length
        silence instead, so STT sees a quiet line and detects the caller's real
        turn cleanly once playback finishes.
        """
        was_muted = False
        voice_on = False
        while True:
            msg = await self._client.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            data = msg.get("bytes")
            if data is not None:
                now = time.monotonic()
                muted = now < self._state.speaking_until + _ECHO_TAIL_SECONDS
                if muted != was_muted:
                    if muted:
                        log.info(
                            "STT muted (agent speaking ~%.1fs)",
                            self._state.speaking_until + _ECHO_TAIL_SECONDS - now,
                        )
                    else:
                        log.info("STT listening (mute released)")
                    was_muted = muted
                # Diagnostic VAD: while listening, log when the caller's audio
                # starts/stops, independent of AssemblyAI. Comparing this trace
                # against the AAI Turn logs tells us whether a quiet stretch is
                # the caller thinking vs. STT failing to transcribe live speech.
                if muted:
                    voice_on = False
                else:
                    samples = np.frombuffer(data, dtype=np.int16)
                    level = (
                        int(np.abs(samples.astype(np.int32)).mean())
                        if samples.size else 0
                    )
                    if level >= _VOICE_LEVEL and not voice_on:
                        voice_on = True
                        log.info("Caller audio started (level=%d)", level)
                    elif level < _VOICE_LEVEL and voice_on:
                        voice_on = False
                        log.info("Caller audio stopped")
                await self._stt.send_audio(bytes(len(data)) if muted else data)
                continue
            # Non-audio frames (browser control messages) are ignored: the
            # pipeline is half-duplex with no barge-in, so there is nothing for
            # the caller to interrupt.

    async def stt_to_agent(self) -> None:
        """Consume STT events; dispatch the agent on a formatted final turn."""
        async for event in self._stt:
            etype = event.get("type")
            if etype == "Begin":
                log.info("AssemblyAI session begin: %s", event)
                continue
            if event.get("error") or etype == "Error":
                err = event.get("error") or event
                log.error("AssemblyAI error message: %s", err)
                await self._client.send_json(
                    {"type": "error", "text": f"AssemblyAI: {err}"}
                )
                continue
            if etype != "Turn":
                log.info("AssemblyAI message: %s", event)
                continue

            transcript = (event.get("transcript") or "").strip()
            end_of_turn = bool(event.get("end_of_turn"))
            log.info(
                "AAI Turn order=%s eot=%s fmt=%s text=%r",
                event.get("turn_order"),
                end_of_turn,
                event.get("turn_is_formatted"),
                transcript,
            )
            if not transcript:
                continue

            if not end_of_turn:
                await self._client.send_json({"type": "partial", "text": transcript})
                # Track when the user last spoke for end-of-turn latency measurement.
                self._state.last_partial_at = time.monotonic()
                continue

            # end_of_turn. Dispatch on the first end-of-turn event without waiting
            # for the punctuated ("formatted") follow-up — the LLM doesn't need
            # punctuation, and waiting for it adds a round trip of latency. Dedupe
            # by turn_order so the later formatted event for the same turn is
            # ignored.
            turn_order = event.get("turn_order", -1)
            if turn_order <= self._state.last_dispatched_turn:
                continue
            self._state.last_dispatched_turn = turn_order

            final_at = time.monotonic()
            user_stopped_at = self._state.last_partial_at
            self._state.last_partial_at = None
            stt_ms = (
                int((final_at - user_stopped_at) * 1000) if user_stopped_at else None
            )

            await self._client.send_json(
                {"type": "final", "text": transcript, "stt_ms": stt_ms}
            )
            log.info(
                "Enqueue turn order=%s qsize=%s text=%r",
                turn_order, self._turns.qsize(), transcript,
            )
            # Real caller turn: play a gap filler while the agent processes it.
            self.enqueue_turn(transcript, user_stopped_at, gap_filler=True)
