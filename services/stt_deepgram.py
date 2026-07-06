"""Deepgram streaming STT wrapper.

Exposes the same interface as services.stt.AssemblyAIStream — connect(),
send_audio(pcm), async iteration yielding event dicts, and close() — so the
pipeline can swap providers without any awareness of which one is live. Deepgram
results are translated into the same {type:"Turn", transcript, end_of_turn,
turn_order, turn_is_formatted} shape AudioBridge already consumes.

Unlike AssemblyAI's semantic end-of-turn model, Deepgram endpoints purely on
silence (the `endpointing` param), so a short reply like a bare "yes" still
finalizes once the caller goes quiet — the failure mode we saw on the phone path
where short affirmations never produced a transcript.
"""
import asyncio
import json
import logging
import time

import websockets

log = logging.getLogger("voice")

# Deepgram model used for transcription. nova-3 is the latest general model and
# transcribes native telephony μ-law@8k (the Twilio path) as well as PCM16@16k
# (the browser path).
_MODEL = "nova-3"

# Milliseconds of trailing silence after speech before Deepgram finalizes the
# current utterance (emits speech_final). Smaller commits short replies faster
# but risks splitting a sentence on a brief pause; tune against real calls.
# Lowered from 900 -> 450: the 900 value predates the composing-window merge
# (TurnHandler keeps a fragment's user message on an early split and the next
# turn reassembles it), which now absorbs most mid-sentence splits — so the
# agent takes its turn ~450 ms sooner.
_ENDPOINTING_MS = 450

# Backstop finalizer: if no speech_final fires, Deepgram emits an UtteranceEnd
# after this much silence between words. Must be >= 1000 (API minimum) and
# requires interim_results. Catches utterances the endpointer misses. Kept >=
# endpointing so it stays a true backstop.
_UTTERANCE_END_MS = 1000

# AudioFormat.stt_encoding values (shared with the AssemblyAI path) mapped to
# Deepgram's encoding names.
_ENCODING_MAP = {"pcm_s16le": "linear16", "pcm_mulaw": "mulaw"}

# Silence fill bytes per encoding, streamed while the agent holds the floor
# (see send_mute). μ-law encodes zero amplitude as 0xFF; PCM16 as 0x00.
_SILENCE_BYTE = {"pcm_mulaw": b"\xff", "pcm_s16le": b"\x00"}

# Mid-call reconnect policy. A phone call must survive the Deepgram socket
# dying (seen live: keepalive ping timeout with no close frame — the connection
# went silently dead and the whole session crashed). The receive loop re-dials
# up to this many times with a growing backoff before giving up.
_RECONNECT_MAX_ATTEMPTS = 3
_RECONNECT_BACKOFF_SECONDS = 0.5

# Outbound audio queue bound (~100 ms frames, so ~3 s of audio). If the uplink
# to Deepgram can't sustain real time, the OLDEST frames are dropped so the
# stream stays near-live: a transcript that lags tens of seconds behind the
# caller (seen live — the agent answered half a minute late and the caller hung
# up) is far worse than a clipped word. The queue also decouples the Twilio
# inbound loop from Deepgram's socket, so a stalled send can never block
# barge-in detection or inbound frame processing.
_SEND_QUEUE_MAX_FRAMES = 30
# Throttle for the dropped-frames warning.
_DROP_LOG_INTERVAL_SECONDS = 2.0


def _dg_url(encoding: str, sample_rate: int) -> str:
    dg_encoding = _ENCODING_MAP.get(encoding, encoding)
    return (
        "wss://api.deepgram.com/v1/listen"
        f"?model={_MODEL}"
        f"&encoding={dg_encoding}&sample_rate={sample_rate}&channels=1"
        "&language=en&interim_results=true&smart_format=true&vad_events=true"
        f"&endpointing={_ENDPOINTING_MS}&utterance_end_ms={_UTTERANCE_END_MS}"
    )


class DeepgramStream:
    def __init__(
        self,
        api_key: str,
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
    ):
        self.api_key = api_key
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.ws = None
        self._closed = False
        # Emitted turn counter. Lives on the instance (not per connection) so it
        # keeps increasing across a mid-call reconnect — AudioBridge dedupes on
        # turn_order, and a reset to 0 would silently drop every turn after the
        # reconnect.
        self._turn_order = 0
        # Bounded outbound queue + sender task (see _SEND_QUEUE_MAX_FRAMES).
        self._send_q: asyncio.Queue[bytes] | None = None
        self._sender: asyncio.Task | None = None
        self._dropped_frames = 0
        self._last_drop_log = 0.0

    async def connect(self):
        # ping_interval=None disables the websockets library's own WS-level
        # ping/pong keepalive. Left on (the default), it PINGs every few seconds
        # and hard-closes with code 1011 "keepalive ping timeout" if a PONG is
        # slow to return — which fired live, tearing down a healthy connection
        # and dumping an uncatchable asyncio traceback from the library's
        # internal keepalive task. We don't need it: audio (or silence fill, see
        # send_mute) flows continuously so Deepgram's no-audio timeout never
        # trips, and a genuinely dead socket still surfaces as a
        # ConnectionClosed on the next read/send, which _events already catches
        # and reconnects.
        self.ws = await websockets.connect(
            _dg_url(self.encoding, self.sample_rate),
            extra_headers={"Authorization": f"Token {self.api_key}"},
            max_size=None,
            ping_interval=None,
        )
        # One sender for the stream's lifetime; it reads whatever socket is
        # current, so a mid-call reconnect doesn't need to restart it.
        if self._sender is None:
            self._send_q = asyncio.Queue(maxsize=_SEND_QUEUE_MAX_FRAMES)
            self._sender = asyncio.create_task(self._send_loop())
        return self

    async def _reconnect(self) -> bool:
        """Re-dial Deepgram after a mid-call socket death. Returns success.

        Owned by the receive loop (`_events`) only, so the send paths never
        race it — they just drop frames onto the dead/absent socket until the
        fresh one is in place (Twilio audio is continuous; a few lost frames
        during the ~1 s re-dial are inaudible to the pipeline).
        """
        old, self.ws = self.ws, None
        if old is not None:
            try:
                await old.close()
            except Exception:  # noqa: BLE001
                pass
        for attempt in range(1, _RECONNECT_MAX_ATTEMPTS + 1):
            if self._closed:
                return False
            try:
                await self.connect()
                log.warning("Deepgram reconnected (attempt %d)", attempt)
                return True
            except Exception as exc:  # noqa: BLE001
                log.warning("Deepgram reconnect attempt %d failed: %s", attempt, exc)
                await asyncio.sleep(_RECONNECT_BACKOFF_SECONDS * attempt)
        return False

    async def send_audio(self, pcm: bytes):
        """Queue a frame for the sender task; never blocks the inbound loop.

        When the queue is full the uplink is behind real time — drop the OLDEST
        frame so what Deepgram hears stays near-live instead of drifting an
        unbounded distance behind the caller.
        """
        q = self._send_q
        if q is None:
            return  # not connected yet
        while True:
            try:
                q.put_nowait(pcm)
                return
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    self._dropped_frames += 1
                except asyncio.QueueEmpty:
                    pass
                now = time.monotonic()
                if now - self._last_drop_log >= _DROP_LOG_INTERVAL_SECONDS:
                    self._last_drop_log = now
                    log.warning(
                        "Deepgram uplink behind real time; dropped %d oldest "
                        "audio frame(s) to stay live",
                        self._dropped_frames,
                    )
                    self._dropped_frames = 0

    async def _send_loop(self):
        """Drain the queue onto whatever socket is currently live."""
        while not self._closed:
            pcm = await self._send_q.get()
            ws = self.ws
            if ws is None:
                continue  # reconnect in progress: drop, audio resumes after
            try:
                await ws.send(pcm)
            except websockets.exceptions.ConnectionClosed:
                # Dead socket. The receive loop detects the same closure and
                # owns the reconnect; raising here would crash the sender.
                pass

    async def send_mute(self, nbytes: int):
        """The agent holds the floor: stream equal-length real-time SILENCE.

        Deepgram paces its streaming decode against the audio timeline it
        receives, so the earlier KeepAlive-pause approach (send nothing while
        muted) put an N-second hole in the stream and everything AFTER the hole
        surfaced ~N seconds late — seen live as transcripts lagging 7-8 s behind
        the caller right after every agent reply, even with a healthy uplink.
        Streaming silence keeps the timeline continuous (and Deepgram's
        endpointing timers running) at real-time cadence.

        The 256 kbps zero-fill burden that originally motivated the pause is
        gone: the phone path now sends native μ-law@8k, so silence costs 8 KB/s.
        On a genuinely constrained uplink the bounded send queue drops oldest
        frames to stay live — and dropped silence is free.
        """
        fill = _SILENCE_BYTE.get(self.encoding, b"\x00")
        await self.send_audio(fill * nbytes)

    def __aiter__(self):
        return self._events()

    async def _events(self):
        # Deepgram streams interim and final result segments continuously. We
        # accumulate final segments into one utterance and emit a single
        # end_of_turn event when the endpointer (speech_final) or the
        # UtteranceEnd backstop fires — giving AudioBridge one turn per reply,
        # with a strictly increasing turn_order for its dedupe.
        #
        # The outer loop is the mid-call reconnect: when the socket dies (seen
        # live as a keepalive ping timeout with no close frame), re-dial and
        # keep the call going instead of ending the stream. Any caller words
        # already finalized on the dying connection are flushed as a turn first
        # so they aren't lost.
        while True:
            connection_events = self._read_connection()
            try:
                async for event in connection_events:
                    yield event
            except websockets.exceptions.ConnectionClosed as exc:
                log.error(
                    "Deepgram closed the stream: code=%s reason=%r",
                    exc.code, exc.reason,
                )
            if self._closed:
                return
            if not await self._reconnect():
                yield {
                    "type": "Error",
                    "error": "Deepgram stream lost and reconnect failed",
                }
                return

    async def _read_connection(self):
        """Yield pipeline events from the current socket until it closes.

        Raises ConnectionClosed to the caller (`_events`) on an abnormal drop,
        after flushing any accumulated final segments as a completed turn.
        """
        final_text = ""
        # Audio-timeline watermark (seconds): end of the newest is_final segment
        # accepted so far. Deepgram occasionally re-emits Results covering audio
        # that was already finalized — seen live as a complete re-transcription
        # of the previous utterance arriving several seconds later, which was
        # then dispatched and answered a second time. A genuine new utterance
        # always advances the timeline, so any segment ending at or before the
        # watermark is a replay and is dropped. Per-connection: a reconnected
        # stream restarts its timeline at zero, so carrying the old watermark
        # over would drop every post-reconnect segment as a "replay".
        finalized_until = 0.0

        def _turn(text: str, end_of_turn: bool, formatted: bool) -> dict:
            return {
                "type": "Turn",
                "transcript": text.strip(),
                "end_of_turn": end_of_turn,
                "turn_order": self._turn_order,
                "turn_is_formatted": formatted,
            }

        try:
            async for raw in self.ws:
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    continue

                mtype = msg.get("type")

                if mtype == "Error" or msg.get("error"):
                    yield {
                        "type": "Error",
                        "error": msg.get("description") or msg.get("error") or msg,
                    }
                    continue

                if mtype == "UtteranceEnd":
                    if final_text:
                        yield _turn(final_text, True, True)
                        final_text = ""
                        self._turn_order += 1
                    continue

                if mtype != "Results":
                    # Metadata / SpeechStarted / etc. — nothing for the pipeline.
                    continue

                alt = (msg.get("channel", {}).get("alternatives") or [{}])[0]
                text = (alt.get("transcript") or "").strip()
                is_final = bool(msg.get("is_final"))
                speech_final = bool(msg.get("speech_final"))
                try:
                    seg_end = float(msg.get("start") or 0.0) + float(
                        msg.get("duration") or 0.0
                    )
                except (TypeError, ValueError):
                    seg_end = 0.0

                if text and 0.0 < seg_end <= finalized_until:
                    log.info(
                        "Deepgram replayed segment (end=%.2fs <= finalized=%.2fs); "
                        "dropped: %r",
                        seg_end, finalized_until, text,
                    )
                    continue

                if not text:
                    # Silence frame. If it carries the endpoint flag, flush any
                    # accumulated final segments as a completed turn.
                    if speech_final and final_text:
                        yield _turn(final_text, True, True)
                        final_text = ""
                        self._turn_order += 1
                    continue

                if is_final:
                    finalized_until = max(finalized_until, seg_end)
                    final_text = (final_text + " " + text).strip()
                    if speech_final:
                        yield _turn(final_text, True, True)
                        final_text = ""
                        self._turn_order += 1
                    else:
                        yield _turn(final_text, False, False)
                else:
                    # Interim hypothesis: show accumulated finals plus the live
                    # guess, but don't commit it to the buffer.
                    yield _turn(f"{final_text} {text}", False, False)
        except websockets.exceptions.ConnectionClosed:
            # Flush what the caller already said (finalized segments that never
            # got their endpoint) as a completed turn so the words survive the
            # drop, then let _events decide whether to reconnect.
            if final_text:
                yield _turn(final_text, True, True)
                self._turn_order += 1
            raise

    async def close(self):
        self._closed = True  # stops any reconnect attempt racing the teardown
        if self._sender is not None:
            self._sender.cancel()
            try:
                await self._sender
            except BaseException:
                pass
            self._sender = None
        if self.ws is not None:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
