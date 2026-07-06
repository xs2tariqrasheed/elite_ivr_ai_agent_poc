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
import json
import logging
import time

import websockets

log = logging.getLogger("voice")

# Deepgram model used for transcription. nova-3 is the latest general model and
# handles 16 kHz telephony audio (upsampled from μ-law) well.
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

# While the agent holds the floor no audio is sent (see send_mute); a KeepAlive
# at this cadence stops Deepgram's 10-second no-audio timeout (NET-0001) from
# closing the stream in the meantime.
_KEEPALIVE_INTERVAL_SECONDS = 3.0


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
        self._last_keepalive = 0.0

    async def connect(self):
        self.ws = await websockets.connect(
            _dg_url(self.encoding, self.sample_rate),
            extra_headers={"Authorization": f"Token {self.api_key}"},
            max_size=None,
            ping_interval=5,
            ping_timeout=20,
        )
        return self

    async def send_audio(self, pcm: bytes):
        if self.ws is not None:
            await self.ws.send(pcm)

    async def send_mute(self, nbytes: int):
        """The agent holds the floor: send NO audio, just a throttled KeepAlive.

        The zero-fill approach (equal-length silence for every muted frame) kept
        Deepgram chewing through 256 kbps of digital zeros for the whole of every
        agent turn. On a constrained uplink that stream falls behind real time,
        and the growing backlog surfaced live as multi-second transcript lag and
        Deepgram re-emitting already-finalized audio (see the replay guard in
        `_events`). Deepgram's documented pattern for "caller shouldn't be heard
        right now" is to pause the audio and send KeepAlive text frames so the
        connection survives its 10 s no-audio timeout — the audio timeline simply
        resumes when real frames flow again after the mute lifts.
        """
        if self.ws is None:
            return
        now = time.monotonic()
        if now - self._last_keepalive >= _KEEPALIVE_INTERVAL_SECONDS:
            self._last_keepalive = now
            await self.ws.send(json.dumps({"type": "KeepAlive"}))

    def __aiter__(self):
        return self._events()

    async def _events(self):
        # Deepgram streams interim and final result segments continuously. We
        # accumulate final segments into one utterance and emit a single
        # end_of_turn event when the endpointer (speech_final) or the
        # UtteranceEnd backstop fires — giving AudioBridge one turn per reply,
        # with a strictly increasing turn_order for its dedupe.
        turn_order = 0
        final_text = ""
        # Audio-timeline watermark (seconds): end of the newest is_final segment
        # accepted so far. Deepgram occasionally re-emits Results covering audio
        # that was already finalized — seen live as a complete re-transcription
        # of the previous utterance arriving several seconds later, which was
        # then dispatched and answered a second time. A genuine new utterance
        # always advances the timeline, so any segment ending at or before the
        # watermark is a replay and is dropped.
        finalized_until = 0.0
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
                        yield {
                            "type": "Turn",
                            "transcript": final_text.strip(),
                            "end_of_turn": True,
                            "turn_order": turn_order,
                            "turn_is_formatted": True,
                        }
                        final_text = ""
                        turn_order += 1
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
                        yield {
                            "type": "Turn",
                            "transcript": final_text.strip(),
                            "end_of_turn": True,
                            "turn_order": turn_order,
                            "turn_is_formatted": True,
                        }
                        final_text = ""
                        turn_order += 1
                    continue

                if is_final:
                    finalized_until = max(finalized_until, seg_end)
                    final_text = (final_text + " " + text).strip()
                    if speech_final:
                        yield {
                            "type": "Turn",
                            "transcript": final_text.strip(),
                            "end_of_turn": True,
                            "turn_order": turn_order,
                            "turn_is_formatted": True,
                        }
                        final_text = ""
                        turn_order += 1
                    else:
                        yield {
                            "type": "Turn",
                            "transcript": final_text.strip(),
                            "end_of_turn": False,
                            "turn_order": turn_order,
                            "turn_is_formatted": False,
                        }
                else:
                    # Interim hypothesis: show accumulated finals plus the live
                    # guess, but don't commit it to the buffer.
                    interim = (final_text + " " + text).strip()
                    yield {
                        "type": "Turn",
                        "transcript": interim,
                        "end_of_turn": False,
                        "turn_order": turn_order,
                        "turn_is_formatted": False,
                    }
        except websockets.exceptions.ConnectionClosed as exc:
            log.error(
                "Deepgram closed the stream: code=%s reason=%r",
                exc.code, exc.reason,
            )
            raise RuntimeError(
                f"Deepgram closed the stream (code {exc.code}): "
                f"{exc.reason or 'no reason given'}"
            ) from exc

    async def close(self):
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
