"""Routes audio between the browser, AssemblyAI STT, and the agent."""
import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from configs.settings import Settings
from services.pipeline_state import PipelineState
from services.stt import AssemblyAIStream

log = logging.getLogger("voice")

# Mean absolute PCM16 amplitude above which an inbound chunk is treated as
# caller speech (vs. line silence / comfort noise) — used only for diagnostic
# voice-activity logging, independent of AssemblyAI. Tune against real calls.
# (Barge-in detection uses its own tunable thresholds; see Settings.)
_VOICE_LEVEL = 400

# Safety cap on the Regime-A pre-roll buffer (frames). The buffer only grows
# during a single consecutive above-threshold run, which fires a barge within
# ~barge_in_min_ms, so it stays small in practice; this just bounds a pathology.
_PREBUFFER_MAX_FRAMES = 50


class AudioBridge:
    """Bridges browser mic audio through STT and dispatches final turns to the agent."""

    def __init__(
        self,
        client: WebSocket,
        stt: AssemblyAIStream,
        state: PipelineState,
        on_turn: Callable[..., Coroutine[Any, Any, None]],
        settings: Settings,
    ) -> None:
        self._client = client
        self._stt = stt
        self._state = state
        self._on_turn = on_turn  # handle_turn(text, user_stopped_at=..., gap_filler=...)
        self._settings = settings
        # Turns are queued and run one at a time. Even with barge-in, only one
        # turn is ever in flight (a barge cancels the current turn before the
        # next is dequeued), so overlapping turns — which would share the agent's
        # memory thread and the single outbound audio stream — never happen.
        self._turns: asyncio.Queue[tuple[str, float | None, bool]] = asyncio.Queue()
        # Barge-in detector state (only used when settings.barge_in_enabled):
        # a debounce run of consecutive above-threshold caller frames, plus a
        # latch so one interruption fires trigger_barge_in exactly once.
        self._barge_run = 0
        self._barge_run_started = 0.0
        self._barge_latched = False
        # Real PCM frames captured during a Regime-A (audible) barge run. STT was
        # fed silence for these, so they're flushed to STT after the barge so the
        # caller's opening words aren't lost from the redo turn (see #3).
        self._barge_prebuffer: list[bytes] = []
        # Throttle for the "barge-watch" diagnostic level log.
        self._barge_log_at = 0.0

    def enqueue_turn(
        self,
        text: str,
        user_stopped_at: float | None = None,
        gap_filler: bool = False,
    ) -> None:
        self._turns.put_nowait((text, user_stopped_at, gap_filler))

    async def turn_worker(self) -> None:
        """Run queued turns strictly in order, one fully finishing before the next.

        A barge-in cancels the in-flight turn task; this worker must survive that
        and loop on to the caller's redo turn, while still dying on a genuine
        teardown cancel (state.closed) so VoiceSession.run's cleanup proceeds.
        """
        while True:
            text, user_stopped_at, gap_filler = await self._turns.get()
            # Fresh turn — re-arm the barge detector (so the redo turn right after
            # an interruption is itself interruptible). The opening greeting is
            # the only turn enqueued with gap_filler=False; exempt it from the
            # composing-window barge so call-start line noise can't cancel it.
            self._barge_run = 0
            self._barge_latched = False
            self._state.greeting_active = not gap_filler
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
                if self._state.closed:
                    raise  # genuine teardown — propagate and end the worker.
                # Barge-in cancel: handle_turn normally catches CancelledError
                # and returns (so we rarely land here), but if a future change
                # makes it re-raise, swallow it and continue to the redo turn.
                log.info("Turn cancelled by barge-in; continuing")
            finally:
                if self._state.turn_task is task:
                    self._state.turn_task = None
                self._state.greeting_active = False
            log.info("Turn worker done: %r", text)

    async def browser_to_stt(self) -> None:
        """Forward mic PCM frames to STT, gating on the agent's playback state.

        Half-duplex core: while the agent's audio is on the wire, equal-length
        silence is sent to STT so the agent's own voice (line/acoustic echo on a
        phone call) can't bleed back in — corrupting transcripts and stalling
        end-of-turn detection.

        With barge-in enabled, an energy detector runs on top of that core so
        sustained caller speech prunes the in-flight turn (see
        `_detect_barge_in` / `trigger_barge_in`). With it disabled the pipeline
        stays strictly half-duplex and the agent cannot be interrupted.
        """
        was_muted = False
        voice_on = False
        while True:
            msg = await self._client.receive()
            if msg.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()
            data = msg.get("bytes")
            if data is None:
                # Non-audio frames (transport control messages): nothing to do.
                continue

            now = time.monotonic()
            echo_tail = self._settings.barge_in_echo_tail_seconds
            audible = now < self._state.speaking_until + echo_tail
            samples = np.frombuffer(data, dtype=np.int16)
            level = int(np.abs(samples.astype(np.int32)).mean()) if samples.size else 0

            if self._settings.barge_in_enabled:
                await self._detect_barge_in(data, level, now, audible)
                continue

            # ---- Legacy strictly-half-duplex path (no barge-in) --------------
            muted = audible
            if muted != was_muted:
                if muted:
                    log.info(
                        "STT muted (agent speaking ~%.1fs)",
                        self._state.speaking_until + echo_tail - now,
                    )
                else:
                    log.info("STT listening (mute released)")
                was_muted = muted
            # Diagnostic VAD: while listening, log when the caller's audio
            # starts/stops, independent of AssemblyAI. Comparing this trace
            # against the AAI Turn logs tells us whether a quiet stretch is the
            # caller thinking vs. STT failing to transcribe live speech.
            if muted:
                voice_on = False
            elif level >= _VOICE_LEVEL and not voice_on:
                voice_on = True
                log.info("Caller audio started (level=%d)", level)
            elif level < _VOICE_LEVEL and voice_on:
                voice_on = False
                log.info("Caller audio stopped")
            await self._stt.send_audio(bytes(len(data)) if muted else data)

    async def _detect_barge_in(
        self, data: bytes, level: int, now: float, audible: bool
    ) -> None:
        """Forward one frame to STT and, while a turn is live, watch for a barge.

        Two regimes, because the agent's reply is buffered whole before any audio
        plays (services/tts.py), so the dominant interrupt window is "agent
        composing, nothing audible yet":
          - REGIME A (audible): agent audio on the wire — keep the echo defense
            (silence to STT) and detect only on energy above a high threshold,
            after a brief onset guard.
          - REGIME B (composing): no audio on the wire, so no echo — feed live
            caller audio to STT and detect at the lower idle threshold.
        A confirmed barge (debounced run) calls `trigger_barge_in` once.
        """
        turn_live = (
            self._state.turn_task is not None and not self._state.turn_task.done()
        )
        # The agent is interruptible while it is composing a reply (turn_live) OR
        # while its buffered audio is still playing out. The latter continues for
        # several seconds AFTER the turn task finishes, because the whole reply is
        # streamed to the transport up front — so gating on turn_live alone left
        # barge-in dead during the tail of every reply (the agent kept talking
        # until its buffer drained). Detect whenever the agent is audible too.
        if not (turn_live or audible):
            # Truly idle: nothing to interrupt; listen for the next utterance.
            self._reset_barge_run()
            self._barge_latched = False
            if level >= _VOICE_LEVEL and now - self._barge_log_at >= 0.5:
                self._barge_log_at = now
                log.info("barge-watch idle level=%d (caller speaking)", level)
            await self._stt.send_audio(data)
            return

        if audible:
            armed = (
                now - self._state.speaking_started_at
                >= self._settings.barge_in_echo_guard_seconds
            )
            threshold = self._settings.barge_in_voice_level
            await self._stt.send_audio(bytes(len(data)))  # silence (echo defense)
        elif self._state.greeting_active:
            # Opening greeting still composing — don't let call-start line noise
            # cancel it before the caller has spoken. Listen, but don't detect.
            self._reset_barge_run()
            await self._stt.send_audio(data)
            return
        else:
            armed = True
            threshold = self._settings.barge_in_voice_level_idle
            await self._stt.send_audio(data)  # live (no echo while composing)

        # Diagnostic heartbeat: while a turn is live, print the measured inbound
        # level against the active threshold so barge sensitivity can be tuned
        # against real calls. Throttled to ~2/sec to avoid log spam.
        if now - self._barge_log_at >= 0.5:
            self._barge_log_at = now
            log.info(
                "barge-watch regime=%s level=%d thr=%d armed=%s run=%d",
                "A" if audible else "B", level, threshold, armed, self._barge_run,
            )

        if self._barge_latched:
            return  # already fired for this interruption; keep forwarding only.
        if armed and level >= threshold:
            if self._barge_run == 0:
                self._barge_run_started = now
            self._barge_run += 1
            if audible:
                # STT was fed silence for this real frame; keep it to replay
                # after the barge so the caller's first words survive (#3).
                self._barge_prebuffer.append(data)
                if len(self._barge_prebuffer) > _PREBUFFER_MAX_FRAMES:
                    self._barge_prebuffer.pop(0)
            elapsed_ms = (now - self._barge_run_started) * 1000
            if (
                self._barge_run >= self._settings.barge_in_min_frames
                and elapsed_ms >= self._settings.barge_in_min_ms
            ):
                self._barge_latched = True
                log.info(
                    "Barge-in detected (regime=%s level=%d thr=%d)",
                    "A" if audible else "B", level, threshold,
                )
                await self.trigger_barge_in()
        elif level < threshold:
            # Run broken by a sub-threshold frame (e.g. an echo spike or click):
            # reset so transients can never accumulate into a barge.
            self._reset_barge_run()

    def _reset_barge_run(self) -> None:
        """Reset the debounce run and discard any captured pre-roll frames."""
        self._barge_run = 0
        self._barge_prebuffer = []

    async def trigger_barge_in(self) -> None:
        """Prune the in-flight turn and flush playback so the caller takes over.

        Order matters: bump the generation FIRST so any outbound chunk that
        loses the race self-suppresses (see TurnHandler._send_audio); flush the
        transport's buffered audio so the caller hears the agent stop within a
        round trip; reset the playback deadline so STT un-mutes immediately;
        drop stale queued turns; then cancel the turn task. We do NOT await the
        task here — this runs on the inbound-audio loop, and the turn worker owns
        the await — so ingestion of the caller's redo is never blocked.
        """
        self._state.barge_generation += 1
        await self._transport_clear()
        self._state.speaking_until = 0.0
        self._state.speaking_started_at = 0.0
        self._drain_turns()
        task = self._state.turn_task
        if task is not None and not task.done():
            task.cancel()
        # Replay the caller's opening frames captured while STT was muted for
        # echo defense (Regime A), now that playback is flushed, so the redo
        # turn isn't missing its leading word.
        if self._barge_prebuffer:
            frames = self._barge_prebuffer
            self._barge_prebuffer = []
            for frame in frames:
                await self._stt.send_audio(frame)

    async def _transport_clear(self) -> None:
        """Flush the transport's buffered outbound audio, serialized with sends."""
        async with self._state.send_lock:
            await self._client.clear()

    def _drain_turns(self) -> None:
        """Drop any queued turns — stale the moment the caller interrupts."""
        while not self._turns.empty():
            try:
                self._turns.get_nowait()
            except asyncio.QueueEmpty:
                break

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
