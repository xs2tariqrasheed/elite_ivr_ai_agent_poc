"""Agent invocation and TTS streaming for a single voice turn."""
import asyncio
import logging
import time
import traceback

from fastapi import WebSocket

from agents.base import VoiceAgent
from configs.settings import Settings
from services.pipeline_state import PipelineState
from services.tts import stream_tts_input

log = logging.getLogger("voice")

# Bytes per second of each TTS wire format, used to project how long a chunk
# will take to play so STT can stay muted for exactly that long (half-duplex).
_BYTES_PER_SECOND = {
    "ulaw_8000": 8000,    # 8-bit μ-law @ 8 kHz
    "pcm_16000": 32000,   # 16-bit PCM @ 16 kHz
}

# Spoken when the agent produces no reply (e.g. the LLM request stalled). TTS
# runs through ElevenLabs, which is independent of OpenAI, so the caller hears
# this and can retry instead of being met with dead silence.
_FALLBACK_REPLY = "Sorry, I didn't catch that. Could you say it again?"


def _pop_sentences(buf: str) -> tuple[list[str], str]:
    """Split completed sentences off the front of `buf`, return (sentences, rest).

    A sentence boundary is sentence-ending punctuation followed by whitespace, so
    a period at the very end of the buffer (which may be mid-token, e.g. "1.")
    is deferred until more text arrives.
    """
    sentences: list[str] = []
    start = 0
    i = 0
    while i < len(buf):
        if buf[i] in ".!?":
            j = i + 1
            while j < len(buf) and buf[j] in ".!?":
                j += 1
            if j < len(buf) and buf[j].isspace():
                sentence = buf[start:j].strip()
                if sentence:
                    sentences.append(sentence)
                start = j
                i = j
                continue
            if j >= len(buf):
                break
        i += 1
    return sentences, buf[start:]


class TurnHandler:
    """Runs the agent → TTS pipeline for one complete user utterance."""

    def __init__(
        self,
        client: WebSocket,
        agent: VoiceAgent,
        settings: Settings,
        state: PipelineState,
        tts_output_format: str = "pcm_16000",
    ) -> None:
        self._client = client
        self._agent = agent
        self._settings = settings
        self._state = state
        self._tts_output_format = tts_output_format

    async def _send_audio(self, chunk: bytes, bps: int) -> None:
        """Send one TTS chunk and advance the half-duplex playback deadline.

        STT stays muted until `speaking_until`, so projecting the playback end
        from the byte count keeps the agent's own audio out of the transcript.
        """
        play_at = time.monotonic()
        if self._state.speaking_until < play_at:
            self._state.speaking_until = play_at
        self._state.speaking_until += len(chunk) / bps
        await self._client.send_bytes(chunk)

    async def _speak(self, text: str, bps: int) -> None:
        """Synthesize and play a fixed line (used for the no-reply fallback)."""
        async def one():
            yield text

        async for chunk in stream_tts_input(
            one(),
            self._settings.elevenlabs_api_key,
            self._settings.elevenlabs_voice_id,
            output_format=self._tts_output_format,
        ):
            await self._send_audio(chunk, bps)

    async def handle_turn(
        self, text: str, user_stopped_at: float | None = None
    ) -> None:
        """Invoke the agent on `text`, then stream the TTS reply to the browser."""
        await self._client.send_json({"type": "speaking_start"})
        pre_ids = await self._agent.checkpoint()

        interrupted = False
        collected: list[str] = []
        stream_failed = False
        bps = _BYTES_PER_SECOND.get(self._tts_output_format, 32000)
        t_start = time.monotonic()
        t_first_token: float | None = None
        try:
            async def reply_sentences():
                nonlocal t_first_token, stream_failed
                buf = ""
                try:
                    async for piece in self._agent.stream_response(text):
                        if t_first_token is None:
                            t_first_token = time.monotonic()
                        collected.append(piece)
                        buf += piece
                        sentences, buf = _pop_sentences(buf)
                        for sentence in sentences:
                            yield sentence
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    # e.g. the OpenAI request timed out after its retries. Flag
                    # it so the turn speaks a fallback rather than going silent.
                    stream_failed = True
                    log.error("LLM stream failed: %s", exc)
                tail = buf.strip()
                if tail:
                    yield tail

            first = True
            async for chunk in stream_tts_input(
                reply_sentences(),
                self._settings.elevenlabs_api_key,
                self._settings.elevenlabs_voice_id,
                output_format=self._tts_output_format,
            ):
                if first:
                    first = False
                    now = time.monotonic()
                    await self._client.send_json({
                        "type": "timings",
                        "agent_ms": (
                            int((t_first_token - t_start) * 1000)
                            if t_first_token
                            else None
                        ),
                        "tts_ms": (
                            int((now - t_first_token) * 1000)
                            if t_first_token
                            else None
                        ),
                        "total_ms": (
                            int((now - user_stopped_at) * 1000)
                            if user_stopped_at
                            else None
                        ),
                    })
                await self._send_audio(chunk, bps)

            reply = "".join(collected).strip()
            log.info("Turn reply complete: %r", reply)
            if not reply:
                # No spoken output — usually the LLM request stalled/timed out.
                # Speak a fallback (via ElevenLabs, independent of OpenAI) so the
                # caller can retry instead of hearing silence. Roll back the
                # turn so the unanswered message doesn't linger in agent memory.
                log.warning("Empty reply (stream_failed=%s); speaking fallback",
                            stream_failed)
                await self._speak(_FALLBACK_REPLY, bps)
                await self._client.send_json({"type": "agent", "text": _FALLBACK_REPLY})
                if not self._state.closed:
                    await self._agent.rollback(pre_ids)
            else:
                await self._client.send_json({"type": "agent", "text": reply})
                snapshot = self._agent.snapshot()
                if snapshot is not None:
                    await self._client.send_json({"type": "state", "state": snapshot})

        except asyncio.CancelledError:
            interrupted = True
            log.info("Turn cancelled (session teardown)")
        except Exception as exc:  # noqa: BLE001
            if self._state.closed:
                log.info("Turn aborted after disconnect")
            else:
                log.error("Agent/TTS error: %s", exc)
                traceback.print_exc()
                try:
                    await self._client.send_json(
                        {"type": "error", "text": f"Agent/TTS error: {exc}"}
                    )
                except Exception:
                    pass

        if interrupted and not self._state.closed:
            await self._agent.rollback(pre_ids)
        try:
            await self._client.send_json({"type": "speaking_end"})
        except Exception:
            pass

        # End of call: if the agent finalized the reservation (step 9), let the
        # closing line finish playing, then hang up by closing the transport.
        # For Twilio, closing the Media Stream ends the <Connect> verb and, with
        # no further TwiML, drops the call.
        if not interrupted and not self._state.closed:
            final_state = self._agent.snapshot()
            if final_state and final_state.get("end_call"):
                remaining = self._state.speaking_until - time.monotonic()
                if remaining > 0:
                    await asyncio.sleep(remaining + 0.5)
                self._state.closed = True
                log.info("End of call — hanging up")
                try:
                    await self._client.close()
                except Exception:
                    pass
